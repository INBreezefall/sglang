"""Read the chosen worker after a real dispatch in both comparison arms."""
import contextvars
import functools
import hashlib
import json
import os
from pathlib import Path
import time

controller = contextvars.ContextVar('route_controller',default=None)
def install(m,folder):
    cls=m.DataParallelController
    # Keep a compact receipt trace only when the actual scheduler reports a new watermark.
    previous={}
    update=m.DPBudget.update_budget
    @functools.wraps(update)
    def budget(self,loads):
        if not getattr(self,'track_dispatches',False):
            return update(self,loads)
        loads=tuple(loads)
        result=update(self,loads)
        changed=[]
        for load in loads:
            rank=load.dp_rank
            seq=getattr(load,'dp_dispatch_ack_seq',None)
            if seq is None or seq<=previous.get(rank,0):continue
            previous[rank]=seq
            changed.append(dict(rank=rank,ack=seq,sent=self.sent_sequences[rank],when=time.time()))
        if changed:
            with (Path(folder)/f'receipts-{os.getpid()}.jsonl').open('a') as f:
                for row in changed:print(json.dumps(row),file=f)
        return result
    m.DPBudget.update_budget=budget
    def wrap(method):
        @functools.wraps(method)
        def select(self,*args,**kwargs):
            token=controller.set(self)
            try:return method(self,*args,**kwargs)
            finally:controller.reset(token)
        return select
    for name in ('total_tokens_scheduler','decode_session_tokens_scheduler'):
        if hasattr(cls,name):setattr(cls,name,wrap(getattr(cls,name)))
    send=m.sock_send
    @functools.wraps(send)
    def record(sock,req,*args,**kwargs):
        val=send(sock,req,*args,**kwargs)
        c=controller.get()
        if c is None or getattr(req,'bootstrap_room',None) is None:return val
        rank=next((i for i,w in enumerate(c.workers) if w is sock),None)
        sid=getattr(req,'session_id',None);ids=getattr(req,'input_ids',None)
        row=dict(room=int(req.bootstrap_room),rank=rank,session_hash=(hashlib.sha256(str(sid).encode()).hexdigest() if sid else None),
                 input_tokens=(len(ids) if ids is not None else None),forced=getattr(req,'routed_dp_rank',None),
                 receipt=getattr(req,'dp_dispatch_seq',None),when=time.time())
        with (Path(folder)/f'dispatch-{os.getpid()}.jsonl').open('a') as f:print(json.dumps(row),file=f)
        return val
    m.sock_send=record
