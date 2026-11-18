"""Keep mutable OpenCode code on a model-gateway-only Docker network."""
import json
import os
from pathlib import Path
import time
import uuid


class GoNetwork:
    def __init__(self,artifacts,upstream_port):
        self.artifacts=Path(artifacts);self.port=upstream_port
        self.owner=uuid.uuid4().hex;self.network=None;self.relay=None;self.client=None
        self.name='autoharness-go-'+self.owner[:12]

    def __enter__(self):
        import docker
        from .fullsource_host import load
        self.client=docker.from_env(timeout=120)
        labels={'autoharness.go-network-owner':self.owner}
        try:
            self.network=self.client.networks.create(self.name,driver='bridge',internal=True,labels=labels)
            self.record={'owner':self.owner,'network_name':self.name,'network_id':self.network.id,
                'relay_name':'autoharness-go-relay-'+self.owner[:12],'upstream_port':self.port}
            self._save()
            self.relay=self.client.containers.run(load()['compiler_image_id'],
                name=self.record['relay_name'],entrypoint='python3',command=['/autoharness-go-relay.py'],
                network='bridge',detach=True,mem_limit='128m',pids_limit=64,cap_drop=['ALL'],
                environment={'AUTOHARNESS_RELAY_HOST':'host.docker.internal','AUTOHARNESS_RELAY_PORT':str(self.port)},
                volumes={str(Path(__file__).with_name('go_relay.py').resolve()):{'bind':'/autoharness-go-relay.py','mode':'ro'}},labels=labels)
            self.network.connect(self.relay,aliases=['go-relay'])
            self.record['relay_id']=self.relay.id;self._save()
            # Invalid authentication must reach the real gateway, while a
            # direct public connection must fail before any model request.
            script="""import socket,urllib.request,urllib.error,time
for attempt in range(30):
 try:
  req=urllib.request.Request('http://go-relay:8080/anthropic/v1/messages',data=b'{}',headers={'Authorization':'Bearer network-selftest-invalid'})
  urllib.request.urlopen(req,timeout=3)
  raise RuntimeError('Invalid key was admitted')
 except urllib.error.HTTPError as error:
  assert error.code==401;break
 except (urllib.error.URLError,TimeoutError,ConnectionError):
  if attempt==29:raise
  time.sleep(.2)
try:
 socket.create_connection(('1.1.1.1',443),timeout=3)
except OSError:pass
else:raise RuntimeError('Direct external egress remains enabled')
print('gateway-only-network-ok')
"""
            checked=self.client.containers.run(load()['compiler_image_id'],entrypoint='python3',command=['-c',script],
                network=self.name,remove=True,cap_drop=['NET_RAW'],mem_limit='128m',pids_limit=32,labels=labels)
            if b'gateway-only-network-ok' not in checked:raise RuntimeError('Private model network self-test failed')
            self.record.update(selftest_passed=True,direct_external_egress_blocked=True,model_requests_in_selftest=0)
            self._save()
            os.environ['AUTOHARNESS_DOCKER_NETWORK']=self.name
            os.environ['GO_GATEWAY_URL']='http://go-relay:8080'
            return self
        except BaseException:
            self.close();raise

    def _save(self):
        self.artifacts.mkdir(parents=True,exist_ok=True)
        (self.artifacts/'network.json').write_text(json.dumps(self.record,indent=2)+'\n')

    def close(self):
        if self.relay is not None:
            self.relay.reload()
            if self.relay.labels.get('autoharness.go-network-owner')!=self.owner:
                raise RuntimeError('Relay ownership changed')
            self.relay.remove(force=True);self.relay=None
        if self.network is not None:
            self.network.reload()
            if self.network.attrs['Labels'].get('autoharness.go-network-owner')!=self.owner:
                raise RuntimeError('Private network ownership changed')
            self.network.remove();self.network=None
        if os.environ.get('AUTOHARNESS_DOCKER_NETWORK')==self.name:
            os.environ.pop('AUTOHARNESS_DOCKER_NETWORK',None)
        if hasattr(self,'record'):
            self.record['cleanup_verified']=True;self._save()
        if self.client is not None:self.client.close();self.client=None

    def __exit__(self,*args):self.close()
