from browser_use.browser.context import BrowserContext
from playwright.async_api._generated import ElementHandle
from typing import Literal, Optional, TypedDict, Generic, TypeVar, Any
from .controllers import (
    check_authorization,
    ensure_url,
    search_email
)
from .signals import (
    UnauthorizedAccess,
    RequireUserConfirmation
)
from pydantic import BaseModel, model_validator
from datetime import datetime, timedelta
from functools import lru_cache
import pickle
import os
import logging
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import email
from email import policy
from email.parser import BytesParser

logger = logging.getLogger(__name__)

_generic_type = TypeVar('_generic_type')
class ResponseMessage(BaseModel, Generic[_generic_type]):
    result: Optional[_generic_type] = None
    error: Optional[str] = None
    success: bool = True

    @model_validator(mode="after")
    def refine_status(self):
        if self.error is not None:
            self.success = False

        return self

class SimpleMailThread(TypedDict):
    id: str
    subject: str
    sender: str
    date: str
    snippet: str
    has_attachment: bool
    
class MailThread(BaseModel):
    id: str
    subject: str
    sender: str
    date: str
    body: str
    
cache_dir = os.path.join('/storages', 'cache')
os.makedirs(cache_dir, exist_ok=True)

@lru_cache(maxsize=256)
def query_cached_element(key: str) -> Optional[Any]:
    """
    Retrieve a cached element handle by its ID.
    This is a placeholder function as caching in Playwright is not straightforward.
    """

    obj_path = os.path.join(cache_dir, key)

    if not os.path.exists(obj_path):
        return None

    with open(obj_path, 'rb') as f:
        element = pickle.load(f)

    return element

def cache_element(key: str, element: Any) -> bool:
    """
    Cache the element handle for later use.
    This is a placeholder function as caching in Playwright is not straightforward.
    """
    
    if not key:
        return False

    obj_path = os.path.join(cache_dir, key)
    if os.path.exists(obj_path):
        return True

    with open(obj_path, 'wb') as f:
        pickle.dump(element, f)

    return True


# 0: get all mails that are currently shown in the screen
async def get_current_threads(
    ctx: BrowserContext,
    silent: bool = False
) -> ResponseMessage[list[SimpleMailThread]]:
    page = await ctx.get_current_page()

    if not page.url.startswith('https://mail.google.com') and silent:
        return ResponseMessage(result=[])

    await ensure_authorized(ctx)

    threads = await page.query_selector_all("tr.zA")  # or div.zA depending on Gmail variant
    results = []

    for i, thread in enumerate(threads):
        thread: ElementHandle  # or div.zA depending on Gmail variant
        subject = await thread.eval_on_selector(".bog", "el => el.innerText")
        sender = await thread.eval_on_selector(".yX.xY .yP", "el => el.innerText")
        snippet = await thread.eval_on_selector(".y2", "el => el.innerText")
        timestamp = await thread.eval_on_selector(".xW.xY span", "el => el.getAttribute('title')")
        has_attachment = await thread.query_selector('[title="Has attachment"]') is not None  # Check for attachment icon
        span_btn = await thread.query_selector('span[@data-thread-id]')  # The span with the date/time 

        if span_btn is None:
            continue

        _id = await span_btn.get_attribute('data-thread-id') or ''  # Get the thread ID from the span element 
        _id = _id.split(':')[-1]

        if not _id or not cache_element(f'mail-thread-{_id}', thread):  # Cache the element for later use
            logger.warning(f"Failed to cache thread element with ID: {_id}")

        # TODO: check this and get infrmation about attachments and labels
        results.append(SimpleMailThread(
            id=_id,  # Placeholder for thread ID, as Gmail's UI does not expose it directly
            subject=subject or "No Subject",
            sender=sender or "Unknown Sender",
            date=timestamp or "Unknown Date",
            snippet=snippet or "",
            has_attachment=has_attachment,  # Placeholder, as attachment info is not available in the thread list
        ))

    return ResponseMessage[list[SimpleMailThread]](result=results)

async def ensure_authorized(ctx: BrowserContext) -> bool:
    if not await check_authorization(ctx):
        await ensure_url(ctx, 'https://mail.google.com/')
        raise UnauthorizedAccess('You are not authorized to access this resource. Please log in to your Google account.')

    await ensure_url(ctx, 'https://mail.google.com/')
    page = await ctx.get_current_page()

    # get all url from href of page
    hrefs = await page.query_selector_all('a[href]')

    for href in hrefs:
        url = await href.get_attribute('href')

        if url and url.startswith('https://'):
            params = parse_qs(urlparse(url).query)

            if 'ik' in params and params['ik']:
                cache_element('ik', params['ik'][0])
                break

    return True

async def craft_query(
    from_date: Optional[str]=None, 
    to_date: Optional[str]=None,
    sender: Optional[str]=None, 
    recipient: Optional[str]=None,
    include_words: Optional[str]="", 
    has_attachment: Optional[bool]=False,
    section: Literal["inbox", "sent", "drafts", "spam", "trash", "starred"] = "inbox"
):
    query_str = f'{include_words}'

    if from_date:
        query_str += f' after:{from_date}'

    if to_date:
        if to_date == from_date:
            to_date_obj = datetime.strptime(to_date, '%Y/%m/%d')
            correct_to_date = to_date_obj + timedelta(days=1)
            to_date = correct_to_date.strftime('%Y/%m/%d')

        query_str += f' before:{to_date}'

    if sender:
        query_str += f' from:{sender}'

    if recipient:
        query_str += f' to:{recipient}'

    if has_attachment:
        query_str += ' has:attachment'

    if section != "inbox":
        if section == 'starred':
            query_str += ' is:starred'
        else:
            query_str += f' in:{section}'

    return query_str.strip()

# 1
async def list_threads(
    ctx: BrowserContext, 
    from_date: Optional[str]=None, 
    to_date: Optional[str]=None,
    sender: Optional[str]=None, 
    recipient: Optional[str]=None,
    include_words: Optional[str]="", 
    has_attachment: Optional[bool]=False,
    section: Literal["inbox", "sent", "drafts", "spam", "trash"] = "inbox"
) -> ResponseMessage[list[SimpleMailThread]]:
    query = await craft_query(
        from_date=from_date, 
        to_date=to_date,
        sender=sender, 
        recipient=recipient,
        include_words=include_words, 
        has_attachment=has_attachment,
        section=section
    )

    if query:
        await search_email(ctx, query)

    return await get_current_threads(ctx, silent=False)

@lru_cache(maxsize=256)
async def read_mail_thread(ctx: BrowserContext, id: str) -> Optional[MailThread]:
    ik = query_cached_element('ik')
    params = {
        'ik': ik,
        'view': 'om',
        'permmsgid': f'msg-f:{id}'
    }
    url = f'https://mail.google.com/mail/u/0/'

    cookies = await ctx.browser_context.cookies(
        url='https://mail.google.com' 
        # Ensure the cookies are fetched from the correct domain
    )

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, 
            params=params, 
            cookies=cookies, 
            headers=headers
        )

    if response.status_code != 200:
        logger.error(f"Failed to fetch mail thread {id}: {response.status_code} {response.text}")
        return None

    soup = BeautifulSoup(response.text, features="html.parser")
    raw = soup.find('pre', {'id': 'raw_message_text'}).text

    msg = BytesParser(policy=policy.default).parsebytes(raw.encode())
    body = msg.get_body(preferencelist=('plain')).get_content() if msg.get_body(preferencelist=('plain')) else ""

    return MailThread(
        id=id,
        subject=msg['subject'] or "No Subject",
        sender=msg['from'] or "Unknown Sender",
        date=str(msg['date'] or "Unknown Date"),
        body=body
    )
    
async def ensure_thread_opened(ctx: BrowserContext, thread_id: str) -> bool:
    xpath = f'div[@data-message-id="#msg-f:{thread_id}"]'

    page = await ctx.get_current_page()
    element = await page.query_selector(xpath)

    if not element:
        target_row = await page.query_selector(f'span[data-thread-id="#thread-f:{thread_id}"]')

    if target_row:
        await target_row.click()

    else:
        element: ElementHandle = await query_cached_element(f'mail-thread-{thread_id}')

        if not element:
            return ResponseMessage[bool](error="E-mail not found", success=False)

        tbody = await page.query_selector('tbody')

        if not tbody:
            await page.goto('https://mail.google.com', wait_until='networkidle')

        tbody = await page.query_selector('tbody')

        if not tbody:
            logger.error("Failed to find tbody element in the page.")
            return ResponseMessage[bool](error="Page broken", success=False)

        # hide the element if it is already visible and append to the tbody
        # TODO: potential bug here: solution: cache the html text only, not the elemnt handle

        await page.evaluate(
             """(tbody, element) => {
                try {
                    if (element.style.display !== 'none') {
                        element.style.display = 'none';
                    }

                    tbody.appendChild(element);
                } catch (e) {
                    console.error('Failed to append:', e);
                }
            }""",
            tbody,
            element
        )

        # refresh the element after appending
        element = await page.query_selector(f'span[data-thread-id="#thread-f:{thread_id}"]')  # or div.zA depending on Gmail variant

        if not element:
            logger.error(f"Element with ID {thread_id} not found after appending.")
            return response_model(error="E-mail not found", success=False)

        await element.click()
        return False
    
    return True


# 2
async def enter_thread(ctx: BrowserContext, thread_id: str) -> ResponseMessage[MailThread]:
    await ensure_authorized(ctx)
    response_model = ResponseMessage[MailThread]

    if not thread_id:
        return response_model(error="Thread ID cannot be empty or n/a.", success=False)

    page = await ctx.get_current_page() 
    await ensure_thread_opened(ctx, thread_id)
    await page.wait_for_selector('div[role="main"]', timeout=5000)
    mail_thread = await read_mail_thread(thread_id)

    direct_url = 'https://mail.google.com/mail/u/0/#inbox/' + page.url.split('/')[-1]
    cache_element(f'direct-url-{thread_id}', direct_url)

    if not mail_thread:
        return response_model(error="Failed to read the mail thread.", success=False)

    return response_model(result=mail_thread)

# 3
async def forward_thread(
    ctx: BrowserContext,
    thread_id: str,
    recipient: str
) -> ResponseMessage[str]:
    await ensure_authorized(ctx)
    response_model = ResponseMessage[str]

    if not thread_id:
        return response_model(error="Thread ID cannot be empty or n/a.", success=False)

# 4
async def reply_to_thread(
    ctx: BrowserContext, 
    thread_id, 
    message: Optional[str] = None
) -> ResponseMessage[str]:
    await ensure_authorized(ctx)
    response_model = ResponseMessage[str]

    if not thread_id:
        return response_model(error="Thread ID cannot be empty or n/a.", success=False)

# 5
async def compose_email(ctx: BrowserContext, to, subject, body):
    pass

# # 6
# async def label_thread(ctx: BrowserContext, thread_id, label: Literal['important', 'starred', 'unread']):
#     pass

# 7 
async def sign_out(ctx: BrowserContext) -> ResponseMessage[bool]:
    response_model = ResponseMessage[bool]

    if not await check_authorization(ctx):
        return response_model(result=True)

    page = await ctx.get_current_page()
    await page.goto('https://mail.google.com/mail/u/0/?logout&hl=en')

    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)

        if os.path.isfile(file_path):
            os.remove(file_path)

    query_cached_element.cache_clear()
    read_mail_thread.cache_clear()

    if 'accounts.google.com' in page.url:
        raise UnauthorizedAccess("Unauthorized access: Google sign-in page detected.")

    return response_model(result=True)