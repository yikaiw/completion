import os
import urllib2

url, material_id, opener = '', '', ''
url = 'http://storage.jd.com' + url
videoname = os.path.basename(url)
os.mkdir(material_id)
videopath = os.path.join(material_id, videoname)
proxy_handler = urllib2.ProxyHandler({'http': 'http://172.22.178.101:80'})
opener = urllib2.build_opener(opener)
with open(videopath, 'wb') as f:
    f.write(urllib2.urlopen(url).read())