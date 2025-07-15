import asyncio
import logging
import os
import json
import httpx
from dotenv import load_dotenv
load_dotenv()

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    Agent,
    AgentThread,
    AsyncFunctionTool,
    AsyncToolSet,
    CodeInterpreterTool,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from agent.stream_event_handler import StreamEventHandler
from utils.terminal_colors import TerminalColors as tc
from utils.utilities import Utilities
from services.mcp_client import CalendarMCPClient

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

AGENT_NAME = "Calendar Scheduler"
FONTS_ZIP = "fonts/fonts.zip"
API_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
PROJECT_CONNECTION_STRING = os.environ["PROJECT_CONNECTION_STRING"]
MAX_COMPLETION_TOKENS = 10240
MAX_PROMPT_TOKENS = 20480
# The LLM is used to generate responses for calendar operations.
# Set the temperature and top_p low to get more deterministic results.
TEMPERATURE = 0.1
TOP_P = 0.1
INSTRUCTIONS_FILE = "../../shared/instructions/calendar_scheduling_instructions.txt"


toolset = AsyncToolSet()
utilities = Utilities()

# Initialize MCP client for calendar operations
mcp_client = CalendarMCPClient()

# Global variable to store shared thread for inter-agent communication
shared_thread_id = None


# MCP wrapper functions for agent tools
async def get_events_via_mcp() -> str:
    """Get events via MCP server."""
    try:
        # Check if MCP server is available
        health = await mcp_client.health_check()
        if not health.get("status") == "healthy":
            return json.dumps({
                "success": False,
                "error": "MCP server not available",
                "message": "Calendar service is currently unavailable"
            })
        
        result = await mcp_client.list_events_via_mcp("all")  # Use "all" as default calendar_id
        if result.get("success"):
            return json.dumps(result)
        else:
            return json.dumps({
                "success": False,
                "error": f"MCP error: {result.get('error')}",
                "message": "Could not retrieve events"
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"MCP connection failed: {e}",
            "message": "Calendar service is currently unavailable"
        })


async def check_room_availability_via_mcp(room_id: str, start_time: str, end_time: str) -> str:
    """Check room availability via MCP server."""
    try:
        health = await mcp_client.health_check()
        if not health.get("status") == "healthy":
            return json.dumps({
                "success": False,
                "error": "MCP server not available",
                "message": f"Cannot check availability for room {room_id}"
            })
        
        result = await mcp_client.check_room_availability_via_mcp(room_id, start_time, end_time)
        if result.get("success"):
            return json.dumps(result)
        else:
            return json.dumps({
                "success": False,
                "error": f"MCP error: {result.get('error')}",
                "message": f"Could not check availability for room {room_id}"
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"MCP connection failed: {e}",
            "message": f"Cannot check availability for room {room_id}"
        })


async def get_rooms_via_mcp() -> str:
    """Get rooms via MCP server."""
    try:
        health = await mcp_client.health_check()
        if not health.get("status") == "healthy":
            return json.dumps({
                "success": False,
                "error": "MCP server not available",
                "message": "Cannot retrieve room list"
            })
        
        result = await mcp_client.get_rooms_via_mcp()
        if result.get("success"):
            return json.dumps(result)
        else:
            return json.dumps({
                "success": False,
                "error": f"MCP error: {result.get('error')}",
                "message": "Could not retrieve room list"
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"MCP connection failed: {e}",
            "message": "Cannot retrieve room list"
        })


async def schedule_event_via_mcp(title: str, start_time: str, end_time: str, 
                               room_id: str, organizer: str, description: str = "") -> str:
    """Schedule event via MCP server."""
    try:
        health = await mcp_client.health_check()
        if not health.get("status") == "healthy":
            return json.dumps({
                "success": False,
                "error": "MCP server not available",
                "message": f"Cannot schedule event '{title}'"
            })
        
        result = await mcp_client.create_event_via_mcp(
            user_id=organizer,
            calendar_id=room_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description
        )
        if result.get("success"):
            return json.dumps(result)
        else:
            return json.dumps({
                "success": False,
                "error": f"MCP error: {result.get('error')}",
                "message": f"Could not schedule event '{title}'"
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"MCP connection failed: {e}",
            "message": f"Cannot schedule event '{title}'"
        })


def fetch_user_directory():
    """Fetch user directory from uploaded Azure resource to verify agent access."""
    url = os.getenv("USER_DIRECTORY_URL")
    if not url:
        print("USER_DIRECTORY_URL not found in environment variables")
        return {}
    
    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        print("Successfully accessed user directory")
        return response.json()
    except Exception as e:
        print(f"Failed to load user directory: {e}")
        return {}


project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=PROJECT_CONNECTION_STRING,
)

# Functions will be defined later after all function definitions

INSTRUCTIONS_FILE = "instructions/general_instructions.txt"


async def add_agent_tools() -> None:
    """Add tools for the agent."""
    font_file_info = None

    # Add the functions tool
    toolset.add(functions)

    # Add the code interpreter tool for data visualization
    code_interpreter = CodeInterpreterTool()
    toolset.add(code_interpreter)

    # Add multilingual support to the code interpreter
    font_file_info = await utilities.upload_file(project_client, utilities.shared_files_path / FONTS_ZIP)
    code_interpreter.add_file(file_id=font_file_info.id)

    # Note: Using static JSON files for calendar data instead of generating synthetic data
    print("Using static calendar data from JSON files")

    return font_file_info


async def initialize() -> tuple[Agent, AgentThread]:
    """Initialize the agent with calendar instructions."""

    if not INSTRUCTIONS_FILE:
        return None, None

    font_file_info = await add_agent_tools()

    try:
        # Load general agent instructions
        instructions = utilities.load_instructions(INSTRUCTIONS_FILE)

        # Inject font file ID if needed
        if font_file_info:
            instructions = instructions.replace("{font_file_id}", font_file_info.id)

        # Diagnostic prints for troubleshooting
        print("API_DEPLOYMENT_NAME:", API_DEPLOYMENT_NAME)
        print("PROJECT_CONNECTION_STRING:", PROJECT_CONNECTION_STRING[:10], "... (hidden)")
        print("Instructions length:", len(instructions))
        print("Toolset:", toolset)

        print("Creating agent...")
        agent = await project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name=AGENT_NAME,
            instructions=instructions,
            toolset=toolset,
            temperature=TEMPERATURE,
        )
        print(f"Created agent, ID: {agent.id}")

        # Test MCP server connectivity
        print("Testing MCP server connectivity...")
        try:
            health = await mcp_client.health_check()
            if health.get("status") == "healthy":
                print("✅ MCP server is healthy and ready")
            else:
                print("⚠️ MCP server health check failed, will fallback to direct service")
        except Exception as e:
            print(f"⚠️ MCP server not reachable: {e}")
            print("📋 Agent will fallback to direct calendar service calls")

        # Test user directory access
        print("Testing user directory access...")
        users = fetch_user_directory()
        if users:
            print(f"📋 User directory loaded successfully with {len(users)} entries")
            # Print first few users for verification (optional)
            for i, (key, value) in enumerate(list(users.items())[:3]):
                print(f"  Sample user {i+1}: {key} -> {value}")
        else:
            print("⚠️ User directory is empty or inaccessible")

        project_client.agents.enable_auto_function_calls(toolset=toolset)
        print("Enabled auto function calls.")

        print("Creating thread...")
        thread = await project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        # Create shared communication thread for inter-agent communication
        shared_thread = await project_client.agents.create_thread()
        global shared_thread_id
        shared_thread_id = shared_thread.id
        print("SHARED_THREAD_ID:", shared_thread_id)

        # Post initialization message to shared thread
        event_payload = {
            "event": "initialized",
            "message": "Calendar agent is now active and ready to schedule events",
            "updated_by": "calendar-agent"
        }

        await project_client.agents.create_message(
            thread_id=shared_thread.id,
            role="user",
            content=json.dumps(event_payload)
        )
        print("📨 Posted initialization message to shared thread.")

        return agent, thread

    except Exception as e:
        logger.error("An error occurred initializing the agent: %s", str(e))
        logger.error("Please ensure you've enabled an instructions file.")
        return None, None


async def cleanup(agent: Agent, thread: AgentThread) -> None:
    """Cleanup the resources."""
    existing_files = await project_client.agents.list_files()
    for f in existing_files.data:
        await project_client.agents.delete_file(f.id)
    await project_client.agents.delete_thread(thread.id)
    await project_client.agents.delete_agent(agent.id)


async def post_message(thread_id: str, content: str, agent: Agent, thread: AgentThread) -> None:
    """Post a message to the Foundry Agent Service."""
    try:
        await project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=content,
        )

        stream = await project_client.agents.create_stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=StreamEventHandler(
                functions=functions, project_client=project_client, utilities=utilities),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            instructions=agent.instructions,
        )

        async with stream as s:
            await s.until_done()
    except Exception as e:
        utilities.log_msg_purple(
            f"An error occurred posting the message: {e!s}")


async def schedule_event_with_notification(title: str, start_time: str, end_time: str, 
                                         room_id: str, organizer: str, description: str = "") -> str:
    """
    Schedule an event via MCP and post notification to shared thread for inter-agent communication.
    """
    # Schedule the event using the MCP client
    result = await schedule_event_via_mcp(title, start_time, end_time, room_id, organizer, description)
    
    # Parse the result to check if event was successfully created
    result_data = json.loads(result)
    
    if result_data.get("success") and shared_thread_id:
        # Get room name for human-readable location
        room_name = f"Room {room_id}"  # Default fallback
        try:
            rooms_result = await get_rooms_via_mcp()
            rooms_data = json.loads(rooms_result)
            for room in rooms_data.get("rooms", []):
                if room.get("id") == room_id:
                    room_name = room.get("name", room_name)
                    break
        except Exception:
            pass  # Use fallback if room lookup fails
        
        # Enhanced event notification payload for communications agent
        event_payload = {
            "event": "created",
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "room_id": room_id,
            "location_name": room_name,  # Human-readable room name
            "organizer": organizer,
            "organizer_email": f"{organizer.lower().replace(' ', '.')}@company.com",  # Generate email format
            "attendees": [f"{organizer.lower().replace(' ', '.')}@company.com"],  # Start with organizer
            "description": description,
            "timezone": "UTC",  # Default timezone
            "notification_required": True,  # Flag for comms agent
            "updated_by": "calendar-agent",
            "event_id": result_data.get("event", {}).get("id"),
            "via_mcp": True  # Flag indicating this came via MCP
        }
        
        try:
            await project_client.agents.create_message(
                thread_id=shared_thread_id,
                role="user",
                content=json.dumps(event_payload)
            )
            print(f"📨 Posted event '{title}' to shared thread (via MCP).")
        except Exception as e:
            print(f"⚠️ Failed to post event to shared thread: {e}")
    
    return result


# Define functions list after all function definitions
functions = AsyncFunctionTool([
    # Basic MCP calendar service functions
    get_events_via_mcp,
    check_room_availability_via_mcp,
    get_rooms_via_mcp,
    # Enhanced booking function with shared thread notifications
    schedule_event_with_notification,
])


async def main() -> None:
    """
    Example questions for SIMPLIFIED booking system:
    - "Show me all available rooms"
    - "Check if the Main Conference Room is available tomorrow at 2pm"
    - "Schedule a meeting in the Alpha Meeting Room for tomorrow at 3pm"
    - "I want to book the Drama Studio for a rehearsal next Friday"
    - "What events are scheduled for this week?"
    """
    async with project_client:
        agent, thread = await initialize()
        if not agent or not thread:
            print(f"{tc.BG_BRIGHT_RED}Initialization failed. Ensure you have uncommented the instructions file for the lab.{tc.RESET}")
            print("Exiting...")
            return

        cmd = None

        while True:
            prompt = input(
                f"\n\n{tc.GREEN}Enter your query (type exit or save to finish): {tc.RESET}").strip()
            if not prompt:
                continue

            cmd = prompt.lower()
            if cmd in {"exit", "save"}:
                break

            await post_message(agent=agent, thread_id=thread.id, content=prompt, thread=thread)

        if cmd == "save":
            print("The agent has not been deleted, so you can continue experimenting with it in the Azure AI Foundry.")
            print(
                f"Navigate to https://ai.azure.com, select your project, then playgrounds, agents playgound, then select agent id: {agent.id}"
            )
        else:
            await cleanup(agent, thread)
            print("The agent resources have been cleaned up.")


if __name__ == "__main__":
    print("Starting async program...")
    asyncio.run(main())
    print("Program finished.")
