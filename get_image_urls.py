import requests
import re
import pandas as pd
from multiprocessing import Pool
import string
import numpy as np

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


def getrandstring():
  letters = string.ascii_letters + string.digits
  letters = list(letters)
  letters = ''.join(np.random.choice(letters,2))
  return letters

def get_image_urls(query_key,nres=50,iter=50):
    """
        Get all image url from google image search
        Args:
            query_key: search term as of what is input to search box.
        Returns:
            (list): list of url for respective images.
 
    """
    urllist = []
    i = 0
    query_key = query_key.replace(' ','+')#replace space in query space with +
    query = query_key
    while (len(urllist)<nres)|(i>iter):
      print('complete: %.2f%%'%(len(urllist)/nres*100))
      tgt_url = 'https://www.google.co.th/search?q={}&tbm=isch&tbs=sbd:0'.format(query)#last part is the sort by relv
      r = requests.get(tgt_url, headers = headers)
      urllist = urllist + [n for n in re.findall('"(https[a-zA-Z0-9_./:-]+.(?:jpg|jpeg|png))",', r.text)]
      urllist = list(set(urllist))
      randomstr = getrandstring()
      query = query_key + randomstr
    print('completed')
    return urllist


class scrapemessenger():
  '''
  input -> page name
  input -> page token
  input -> npool (amount of CPU threads for multiprocessing)
  '''
  def __init__(self,pagename,pagetoken,npool):
    self.pagename = pagename
    self.pagetoken = pagetoken
    self.api = facebook.GraphAPI(pagetoken)
    self.pool = Pool(npool)
  def get_page_id(self):
    self.pageid = self.api.get_object(self.pagename)['id']
  def get_threadlist(self,npage):
    content = self.api.get_object(self.pageid+'/conversations')
    cdat = content['data']
    tlist = [cdat[i]['id'] for i in range(len(cdat))]
    i = 0
    while ('paging' in content.keys()) & (i<npage):
      print('fetch page %d'%i)
      if 'next' in content['paging'].keys():
        nurl = content['paging']['next']
        content = requests.get(nurl).json()
        if 'data' in content.keys():
          cdat = content['data']
          tlist = tlist + [cdat[i]['id'] for i in range(len(cdat))]
      else:
        content = {}
        break
      i = i+1
    self.tlist = tlist
  def get_all_message_ids(self):
    tlist = [(i,self.tlist[i],self.api) for i in range(len(self.tlist))]
    #p = Pool(npool)
    mlist = self.pool.map(self.get_idmessages,tlist)
    self.midList = [m for m in mlist if m != None]
  def getalldata(self):
    mtlist = [(i,self.midList[i],self.api) for i in range(len(self.midList))]
    #p = Pool(npool)
    self.messagedat = pd.concat(self.pool.map(self.getmsglist,mtlist),axis=0)
  
  @staticmethod
  def get_idmessages(itid):
    #self = itid[0]
    try:
      api = itid[2]
      tid = itid[1]
      i = itid[0]
      print(i)
      args = {'fields':'message'}
      #return tid
      content = api.get_object(tid+'/messages',**args)
      cdat = content['data']
      mlist = [c['id'] for c in cdat]
      res = {}
      res['tid'] = tid
      res['midlist'] = mlist
      return res
    except:
      pass

  @staticmethod
  def getmsglist(itid):
    try:
      i = itid[0]
      print(i)
      tidm = itid[1]
      tid = tidm['tid']
      mList = tidm['midlist']
      api = itid[2]
      tconver = []
      for mid in mList:
        args = {'fields':'to,from,created_time,message,tags'}
        dat = api.get_object(mid,**args)
        #dat = get_message_meta(mid)
        ct = dat['created_time']
        fid = dat['from']['id']
        fname = dat['from']['name']
        msg = dat['message']
        toid = dat['to']['data'][0]['id']
        tname = dat['to']['data'][0]['name']
        res = [msg,tid,fid,toid,fname,tname,ct]
        if msg!='':
          tconver.append(res)
      tconver.reverse()
      res = pd.DataFrame(data=tconver,columns=['message','threadid','fromid','toid','fromname','toname','datetime'])
      return res
    except:
      pass
    
  def scrapemessages(self,npage):
    print('step 1/4 get page id')
    self.get_page_id()
    print('step 2/4 get conversation thread_ids')
    self.get_threadlist(npage)
    print('step 3/4 get message_ids from thread_ids')
    self.get_all_message_ids()
    print('step 4/4 get messages from message_ids')
    self.getalldata()
    return self.messagedat

class scrapewall():
  '''
  input -> page name
  input -> page token
  input -> npool (amount of CPU threads for multiprocessing)
  '''
  def __init__(self,pagename,pagetoken,npool):
    self.pagename = pagename
    self.pagetoken = pagetoken
    self.api = facebook.GraphAPI(pagetoken)
    self.pool = Pool(npool)
  def get_page_id(self):
    self.pageid = self.api.get_object(self.pagename)['id']
  @staticmethod
  def get_postlist(self,npage):
    resList = []
    args = {'fields':'from,message,likes.summary(true),shares'}
    res = self.api.get_object(self.pageid+'/posts',**args)
    #return res
    i = 0
    while ('next' in res['paging'].keys()) & (i<npage):
        nurl = res['paging']['next']
        res = requests.get(nurl).json()
        #return res
        print('page '+str(i))
        i = i + 1
        resdat = res['data']
        resList = resList + [resdat[i] for i in range(len(resdat))]
    self.postlist = resList
  @staticmethod
  def get_all_postcomment(self):
    plist = [(self.api,self.postlist[i],i) for i in range(len(self.postlist))]
    #return plist
    commentlist = self.pool.map(self.get_postcomment,plist)
    #return commentlist
    self.postdat = pd.concat([pd.DataFrame.from_dict(c) for c in commentlist if c != None],axis=0)

  @staticmethod
  def get_postcomment(postitem):
    try:
      #return postitem
      api = postitem[0]
      pst = postitem[1]
      pmessage = pst['message']
      pid = pst['id']
      likecount = pst['likes']['summary']['total_count']
      sharecount = pst['shares']['count']
      i = postitem[2]
      print(i)
      cdic = {}
      cdic['pid'] = pid
      cdic['postmessage'] = pmessage
      cdic['likecount'] = likecount
      cdic['sharecount'] = sharecount
      res = api.get_object(pid+'/comments')
      #return res
      dList = res['data']
      cres = []
      ctim = []
      cid = []
      for d in dList:
        cres.append(d['message'])
        ctim.append(d['created_time'])
        cid.append(d['id'])
      cdic['message'] = cres
      cdic['created_time'] = ctim
      cdic['comment_id'] = cid
      return cdic
    except:
      pass
  def getpostdata(self,npage):
    print('step 1/3 get page id')
    self.get_page_id()
    print('step 2/3 get post id')
    self.get_postlist(self,npage)
    print('step 3/3 get comments from posts')
    self.get_all_postcomment(self)
    return self.postdat


class twitter_scrape():
  '''
    credential = {}
    credential['consumer_key'] = xxx
    credential['consumer_secret'] = xxx
    credential['access_token'] = xxx
    credential['access_token_secret'] = xxx
  '''
  def __init__(self,credential):
    authen = tweepy.OAuthHandler(credential['consumer_key'], credential['consumer_secret'])
    authen.set_access_token(credential['access_token'], credential['access_token_secret'])
    self.api = tweepy.API(authen)
    self.data = pd.DataFrame()

  def querydata(self,query):
    alldat = self.api.search(q=query,lang='th',count=100)
    if len(alldat)==0:
      print('search not found')
      return 'search not found'
    allres = []
    for dat in alldat:
      result = {}
      result['id'] = dat.id
      result['created_at'] = dat.created_at
      result['text'] = dat.text
      ht = dat.entities['hashtags']
      htag = [h['text'] for h in ht]
      try:
        md = dat.extended_entities['media']
        iurl = [m['media_url_https'] for m in md]
      except:
        iurl = []
      result['image'] = iurl 
      result['hashtag'] = htag
      result['retweet_count'] = dat.retweet_count
      result['author_name'] = dat.author.name
      result['author_id'] = dat.author.id
      allres.append(result)
    res = pd.DataFrame(allres)
    res = res.sort_values(by='retweet_count',ascending=False)
    dat = pd.concat([self.data,res],axis=0)
    self.data = dat.sort_values(by='retweet_count',ascending=False)
    return res

  def gettrenddata(self,woeid=1225448):
    tdat = self.api.trends_place(woeid)
    tdat = tdat[0]['trends']
    tdat = [t['name'] for t in tdat]
    tresList = []
    for t in tdat:
      print(t)
      tres = self.querydata(t)
      tres['query'] = t
      tresList.append(tres)
    res = pd.concat(tresList,axis=0)
    res = res.sort_values(by='retweet_count',ascending=False)
    return res

  def getimages(self):
    imgList = self.data['image'].values
    allimg = []
    for i in imgList:
      allimg = allimg + i
    return allimg

  def gethashtag(self):
    tagList = self.data['hashtag'].values
    allhashtag = []
    for i in tagList:
      allhashtag = allhashtag + i
    return allhashtag














