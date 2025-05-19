from browser_use import Agent
import logging
from .signals import UnauthorizedAccess


logger = logging.getLogger()

async def on_step_start(agent: Agent) -> Agent: 

    return agent

async def on_step_end(agent: Agent) -> Agent:
    current_page = await agent.browser_context.get_current_page()
    
    if 'accounts.google.com/v3/signin/identifier' in current_page.url:
        raise UnauthorizedAccess("Unauthorized access: Google sign-in page detected.") 

    return agent
