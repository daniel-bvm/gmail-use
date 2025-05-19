from browser_use import Controller, Browser, ActionResult
from .models import browser_use_custom_models
from browser_use.browser.context import BrowserContext
from .signals import (
    UnauthorizedAccess,
    RequireUserConfirmation
)
from typing import Literal
from fnmatch import fnmatch

built_in_actions = [
    'done',
    'search_google',
    'go_to_url',
    'go_back',
    'wait',
    'click_element_by_index',
    'input_text',
    'save_pdf',
    'switch_tab',
    'open_tab',
    'close_tab',
    'extract_content',
    'scroll_down',
    'scroll_up',
    'send_keys',
    'scroll_to_text',
    'get_dropdown_options',
    'select_dropdown_option',
    'drag_drop',
    'get_sheet_contents',
    'select_cell_or_range',
    'get_range_contents',
    'clear_selected_range',
    'input_selected_cell_text',
    'update_range_contents' 
]

_controller = Controller(
    output_model=browser_use_custom_models.FinalAgentResult,
    exclude_actions=[
        a
        for a in built_in_actions
        if a not in [
            'done',
            # 'search_google',
            'go_to_url',
            'go_back',
            'wait',
            'click_element_by_index',
            'input_text',
            # 'save_pdf',
            # 'switch_tab',
            # 'open_tab',
            # 'close_tab',
            'extract_content',
            'scroll_down',
            'scroll_up',
            'send_keys',
            # 'scroll_to_text',

            'get_dropdown_options',
            'select_dropdown_option',

            # 'drag_drop',
            # 'get_sheet_contents',
            # 'select_cell_or_range',
            # 'get_range_contents',
            # 'clear_selected_range',
            # 'input_selected_cell_text',
            'update_range_contents' 
        ]
    ]
)


async def check_authorization(ctx: BrowserContext) -> bool:
    site = 'accounts.google.com'
    cookies = await ctx.session.context.cookies()

    for cookie in cookies:
        if not hasattr(cookie, 'domain') or not hasattr(cookie, 'name') or not hasattr(cookie, 'value'):
            continue

        if cookie.domain == site and cookie.name == 'SID' and cookie.value:
            return True

    return False

async def ensure_url(ctx: BrowserContext, pat: str) -> None:
    page = await ctx.get_current_page()
    current_url = page.url

    if not fnmatch(current_url, pat):
        await page.goto(pat, wait_until='networkidle')

@_controller.action('Sign out Google account')
async def sign_out(browser: BrowserContext):
    sites = ['accounts.google.com', 'mail.google.com']

    for site in sites:
        await browser.session.context.clear_cookies(domain=site)
        
    page = await browser.get_current_page()
    await page.reload(wait_until='networkidle')

    return ActionResult(extracted_content='Sign out successful!')

@_controller.action('Open the user mail box')
async def open_mail_box(browser: BrowserContext):
    page = await browser.get_current_page()
    
    await page.goto(
        'https://mail.google.com/mail/u/0/', 
        wait_until='networkidle' # networkidle
    )

    return ActionResult(extracted_content='Navigated to input box')

@_controller.action('Craft a new email')
async def fill_email_form(browser: BrowserContext, subject: str, body: str, recipient: str):
    page = await browser.get_current_page()

    await page.fill('input[aria-label="To recipients"]', recipient)
    await page.fill('input[name="subjectbox"]', subject)
    await page.fill('div[aria-label="Message Body"]', body)

    return ActionResult(extracted_content='Email form filled successfully!')

@_controller.action('Search email')
async def search_email(browser: BrowserContext, query: str):
    page = await browser.get_current_page()

    # Wait for the search box to be available
    await page.wait_for_selector('input[name="q"]', timeout=5000)
    await page.fill('input[name="q"]', query)
    await page.click('button[aria-label="Search mail"]')
    await page.wait_for_timeout(2000)

    return ActionResult(extracted_content='Search executed successfully!')

def get_controler():
    global _controller
    return _controller