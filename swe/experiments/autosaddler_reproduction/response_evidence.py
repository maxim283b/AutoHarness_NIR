"""Unintegrated diagnostic candidate: retain decoded response bytes before forwarding.

No HTTP dispatch, billing settlement, retry, model selection, or production hook.
EOF records transport observation only; it does not certify a valid model result.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


class ResponseEvidence:
    def __init__(self, directory, request_id):
        if not re.fullmatch('[0-9a-f]{32}', request_id):
            raise ValueError('Expected authoritative 32-hex request ID')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.body_path=self.directory/(request_id+'.decoded_body')
        self.meta_path=self.directory/(request_id+'.json')
        self.body=None
        self.state=dict(request_id=request_id,started_at_utc=datetime.now(timezone.utc).isoformat(),
            http_status=None,content_type=None,transport_eof=False,body_bytes=0,
            billing_status='not_evaluated',model_result_validity='not_evaluated',
            retry_performed=False,bytes_representation='HTTPX decoded response bytes, not encrypted wire bytes')

    def __enter__(self):
        # Exclusive creation prevents overwriting a previous request's evidence.
        with self.meta_path.open('x',encoding='utf-8') as f:
            json.dump(self.state,f);f.flush();os.fsync(f.fileno())
        self.body=self.body_path.open('xb')
        return self

    def headers(self,status,content_type):
        # Never accepts a request-header dictionary, credentials, or full headers.
        if type(status) is not int or not 100<=status<=599:raise ValueError('Invalid HTTP status')
        self.state.update(http_status=status,content_type=str(content_type)[:128])

    def append_before_forward(self,chunk):
        if not isinstance(chunk,bytes):raise TypeError('Expected decoded bytes')
        self.body.write(chunk);self.body.flush();os.fsync(self.body.fileno())
        self.state['body_bytes']+=len(chunk)

    def chunks(self,iterator):
        for chunk in iterator:
            self.append_before_forward(chunk)
            yield chunk
        self.state['transport_eof']=True

    def __exit__(self,kind,error,traceback):
        if self.body is not None:self.body.close()
        self.state.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error_type=None if kind is None else kind.__name__,
            decoded_body_sha256=hashlib.sha256(self.body_path.read_bytes()).hexdigest())
        tmp=self.meta_path.with_suffix('.json.tmp')
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(self.state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.meta_path)
        return False
