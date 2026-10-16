"""Colab CLI with 60s minimum HTTP timeout; leaves shared installation untouched."""
import requests
original=requests.sessions.Session.request
def request(self,*args,**kwargs):
 t=kwargs.get('timeout')
 if isinstance(t,(int,float)) and t<60:kwargs['timeout']=60
 return original(self,*args,**kwargs)
requests.sessions.Session.request=request
from colab_cli.cli import main
main()
