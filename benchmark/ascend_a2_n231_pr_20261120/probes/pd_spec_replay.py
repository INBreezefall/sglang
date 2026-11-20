"""Run the fixed replay on fresh HTTP connections, preserving its workload, timing and no-retry behavior."""
from functools import partial
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import aiohttp

output=Path(sys.argv[sys.argv.index('--output')+1])
connector=partial(aiohttp.TCPConnector,force_close=True)
with patch.object(aiohttp,'TCPConnector',connector):
    runpy.run_path(str(Path(__file__).with_name('pd_client.py')),run_name='__main__')
result=json.loads(output.read_text())
result['http_connections']='fresh per HTTP request, no retries'
output.write_text(json.dumps(result,indent=2)+'\n')
