from zeep import Transport, Client
from zeep.cache import SqliteCache
from requests import session

session = session()
cache = SqliteCache(path='/tmp/zeep-cache.db')
transport = Transport(cache=cache, session=session)
client = Client('http://www.dneonline.com/calculator.asmx?WSDL', transport=transport)
print(Transport)
print(client)