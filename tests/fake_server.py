#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self,fmt,*args):return
 def do_GET(self):
  if self.path=="/health":self.send_response(200);self.end_headers();self.wfile.write(b"ok")
  else:self.send_response(404);self.end_headers()
 def do_POST(self):
  n=int(self.headers.get("Content-Length","0"));self.rfile.read(n);body=json.dumps({"choices":[{"message":{"role":"assistant","content":"OK"}}]}).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
def main():
 p=argparse.ArgumentParser();p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,required=True);a=p.parse_args();ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=="__main__":main()
