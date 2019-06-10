from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


from multiprocessing.dummy import Pool as ThreadPool
import time
import os
c=0

def image_scrapper(obj):
    req = Request(obj['link'],headers={'User-Agent' : "Magic Browser"})
    try:
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage)
        img = soup.find("meta",  property="og:image")
        if img is None:
            obj['image'] = ''
        else:
            omg = img['content']
            if not os.path.isfile("images/"+omg.split('/')[-1].split('?')[0]):
                resource = urlopen(img['content'].split('?')[0]+"?ops=600_600")
                output = open("images/"+omg.split('/')[-1].split('?')[0],"wb")
                output.write(resource.read())
                output.close()
                obj['image'] = omg.split('/')[-1].split('?')[0]
        
    except Exception as inst:
        print(inst, obj['headline'])

pool = ThreadPool(16)
start = time.time()
results = pool.map(image_scrapper, data)
pool.close()
pool.join()
end = time.time()

print(end -start)

# print result