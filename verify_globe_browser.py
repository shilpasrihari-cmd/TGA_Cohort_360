"""Manual browser check: python verify_globe_browser.py URL [--dashboard]."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
    page = browser.new_page(viewport={'width': 1400, 'height': 1100})
    errors = []
    external_requests = []
    page.on('request', lambda request: external_requests.append(request.url) if request.url.startswith(('http:', 'https:')) and not request.url.startswith(sys.argv[1].rstrip('/')) else None)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(sys.argv[1])
    page.locator('iframe').first.wait_for(timeout=30000)
    frame = page.frames[-1]
    frame.wait_for_function('window.globeReady === true', timeout=30000)
    assert frame.evaluate('markerData.reduce((n,d)=>n+d.count,0)') == 480
    assert frame.evaluate('globe.globeMaterial().map.image.naturalWidth') == 3600
    assert frame.evaluate('getComputedStyle(document.body).backgroundColor') == 'rgb(6, 13, 27)'
    canvas = frame.locator('canvas').first
    box = canvas.bounding_box()
    diameter = frame.evaluate('''()=>{
      const r=globe.getGlobeRadius(),d=globe.camera().position.length();
      const f=globe.height()/(2*Math.tan(globe.camera().fov*Math.PI/360));
      return 2*f*r/Math.sqrt(d*d-r*r);
    }''')
    assert 550 <= diameter <= 600, diameter
    assert .80 <= diameter / box['height'] <= .85
    frame.locator('.marker[title^="India:"]').click()
    frame.wait_for_timeout(1200)
    focused = frame.evaluate('globe.pointOfView()')
    assert abs(focused['lat']-20.593684)<.1 and abs(focused['lng']-78.96288)<.1, focused
    assert focused['altitude'] < frame.evaluate('defaultView.altitude')
    assert 'India' in frame.locator('#selection').inner_text()
    page.mouse.click(box['x']+15,box['y']+70)
    frame.wait_for_timeout(1200)
    reset = frame.evaluate('globe.pointOfView()')
    assert abs(reset['lng']-40)<.1 and abs(reset['lat']-22)<.1, reset
    assert frame.locator('#selection').is_hidden()
    before = frame.evaluate('globe.pointOfView()')
    x,y = box['x'] + box['width']/2, box['y'] + box['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+130,y+45,steps=15);page.mouse.up()
    page.wait_for_timeout(300)
    after = frame.evaluate('globe.pointOfView()')
    assert abs(after['lng']-before['lng']) > 1, (before, after)
    frame.wait_for_timeout(1200)
    assert abs(frame.evaluate('globe.pointOfView().lng')-after['lng']) < .1, 'Drag unexpectedly reset the camera'
    page.mouse.wheel(0,-300);page.wait_for_timeout(300)
    zoom = frame.evaluate('globe.pointOfView()')
    assert abs(zoom['altitude']-after['altitude']) > .01
    frame.locator('#reset').click()
    frame.wait_for_timeout(1200)
    assert abs(frame.evaluate('globe.pointOfView().altitude-defaultView.altitude')) < .01
    assert abs(frame.evaluate('globe.pointOfView().lng')-40) < .1
    print(f'PASS: desktop diameter {diameter:.1f}px; marker zoom, background reset, drag protection, Reset Globe')
    if '--dashboard' in sys.argv:
        from data import synthetic_data
        people, _ = synthetic_data()
        page.get_by_role('combobox').nth(1).click()
        page.get_by_role('option', name='India', exact=True).click()
        page.keyboard.press('Escape')
        page.wait_for_timeout(1000)
        frame = page.frames[-1]
        frame.wait_for_function('window.globeReady === true && markerData.length === 1', timeout=30000)
        expected = int((people.country == 'India').sum())
        assert frame.evaluate('markerData[0].count') == expected
        assert str(expected) in page.locator('[data-testid="stMetricValue"]').first.inner_text()
        assert frame.locator('.marker').first.get_attribute('title') == f'India: {expected} synthetic participants'
        page.get_by_role('button', name='Reset filters').click()
        page.wait_for_timeout(1000)
        page.get_by_role('combobox').nth(0).click()
        page.get_by_role('option', name='Europe', exact=True).click()
        page.keyboard.press('Escape')
        page.wait_for_timeout(1000)
        frame = page.frames[-1]
        frame.wait_for_function('window.globeReady === true && markerData.length === 3', timeout=30000)
        total = frame.evaluate('markerData.reduce((n,d)=>n+d.count,0)')
        assert total == int((people.region == 'Europe').sum())
        assert str(total) in page.locator('[data-testid="stMetricValue"]').first.inner_text()
        page.get_by_role('tab', name='Learning Intelligence', exact=False).click()
        assert page.get_by_text('Learning vs. support questions', exact=True).is_visible()
        page.get_by_role('tab', name='AI Insights & Recommendations', exact=False).click()
        assert page.get_by_text('Suggested Follow-up Topics', exact=True).is_visible()
        page.get_by_role('tab', name='Global Community', exact=False).click()
        print('PASS: India and Europe filters update markers and reconcile with KPIs; both other tabs render')
    page.set_viewport_size({'width': 390, 'height': 1100})
    page.wait_for_timeout(500)
    mobile_diameter = frame.evaluate('''()=>{
      const r=globe.getGlobeRadius(),d=globe.camera().position.length();
      const f=globe.height()/(2*Math.tan(globe.camera().fov*Math.PI/360));
      return 2*f*r/Math.sqrt(d*d-r*r);
    }''')
    assert mobile_diameter < frame.evaluate('globe.width()')
    assert abs(mobile_diameter/frame.evaluate('globe.width()')-.83) < .02
    page.set_viewport_size({'width': 1400, 'height': 1100})
    page.wait_for_timeout(500)
    print('PASS: responsive globe fits a 390px browser viewport')
    frame.locator('canvas').first.screenshot(path=str(Path(__file__).parent / 'assets' / 'globe' / 'verified-preview.png'))
    assert not errors, errors
    assert not external_requests, external_requests
    print('PASS: texture 3600px, dark background, 480 participants, mouse rotation, wheel zoom, no JS errors or third-party requests')
    browser.close()
