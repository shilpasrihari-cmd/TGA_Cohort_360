"""Browser verification of aggregate Learning Intelligence; URL is a local app."""
import sys
import pandas as pd
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
    page = browser.new_page(viewport={'width':1400,'height':1100}, accept_downloads=True)
    errors, external = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: external.append(request.url) if request.url.startswith(('http:', 'https:')) and not request.url.startswith(sys.argv[1].rstrip('/')) else None)
    page.goto(sys.argv[1])
    page.get_by_role('tab', name='Learning Intelligence', exact=False).click()
    page.get_by_text('Most frequently discussed AI topics', exact=True).wait_for(timeout=60000)
    metrics = page.locator('[data-testid="stMetricValue"]:visible')
    assert metrics.nth(0).inner_text() == '176'
    assert page.get_by_text('Most frequently discussed AI topics', exact=True).is_visible()
    assert page.get_by_role('tab').count() == 3
    assert not page.get_by_text('AI-Discovered Question Themes', exact=True).is_visible()
    page.get_by_role('tab', name='AI Insights & Recommendations', exact=False).click()
    assert page.get_by_text('AI-Discovered Question Themes', exact=True).is_visible()
    assert page.get_by_text('Suggested Follow-up Topics', exact=True).is_visible()
    page.get_by_text('AI Model Evaluation', exact=True).wait_for(timeout=60000)
    assert page.get_by_text('AI Model Evaluation', exact=True).count() == 1
    page.get_by_text('76 model-pipeline question predictions require human review in the full fixture.', exact=True).wait_for(timeout=60000)
    assert page.get_by_text('AI-Discovered Question Themes', exact=True).count() == 1
    # Validate the actual user-facing download, not just the export function.
    with page.expect_download() as download_info:
        page.get_by_role('button', name='Download synthetic aggregate themes').click()
    download = download_info.value
    frame = pd.read_csv(download.path())
    assert set(frame.columns) == {'data_status','theme','topic','question_count','participant_count','needs_review'}
    assert (frame.participant_count >= 3).all()
    assert (frame.data_status == 'Synthetic').all()
    page.get_by_role('tab', name='Learning Intelligence', exact=False).click()
    page.get_by_role('combobox', name='AI topic classifications', exact=True).click()
    page.get_by_role('combobox', name='AI topic classifications', exact=True).fill('Course Support')
    page.get_by_role('option', name='Course Support', exact=True).click()
    page.keyboard.press('Escape')
    page.wait_for_timeout(1000)
    assert metrics.nth(0).inner_text() == '28'
    assert metrics.nth(1).inner_text() == '0'
    page.get_by_role('button', name='Reset filters').click()
    page.wait_for_timeout(1000)
    assert metrics.nth(0).inner_text() == '176'
    page.get_by_role('tab', name='AI Insights & Recommendations', exact=False).click()
    assert page.get_by_text('Suggested Follow-up Topics', exact=True).is_visible()
    page.get_by_role('tab', name='Global Community', exact=False).click()
    assert metrics.nth(0).inner_text() == '60'
    page.locator('iframe').first.wait_for()
    globe_frame = page.frames[-1]
    globe_frame.wait_for_function('window.globeReady === true', timeout=30000)
    assert globe_frame.evaluate('markerData.reduce((sum,d)=>sum+d.count,0)') == 60
    assert globe_frame.evaluate("markerData.every(d=>globe.pointColor()(d)==='#FF0000')")
    assert not errors, errors
    assert not external, external
    print('PASS: local-model Learning Intelligence displays 176 questions; topic filter yields 28 course-support questions; aggregate download suppresses small groups; reset and other tabs work; globe remains red; no third-party requests or JS errors')
    browser.close()
