# Night globe assets

Earth texture: NASA Earth Observatory, **Night Lights 2012 Map**, Suomi NPP / VIIRS, 3600 × 1800 pixels.

- Source: https://science.nasa.gov/earth/earth-observatory/night-lights-2012-map-79765/
- Original image: https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg
- Usage: https://www.nasa.gov/nasa-brand-center/images-and-media/

NASA generally permits educational and informational use of its imagery with attribution. This image is used as a historical illustrative texture; it does not represent current lighting or participant locations. No NASA endorsement is implied. The downloaded image is unmodified.

JavaScript: Globe.GL **2.45.0**, https://github.com/vasturiano/globe.gl, MIT license. Downloaded prebuilt UMD bundle: https://unpkg.com/globe.gl@2.45.0/dist/globe.gl.min.js. Three.js and supporting libraries are included in that bundle. See LICENSE-globe.gl.txt and bundled third-party notices.

The globe sends no external requests at runtime: the script is inline, the image is a data URI, and the iframe CSP disallows network connections. The ordinary dashboard is served by Streamlit.
