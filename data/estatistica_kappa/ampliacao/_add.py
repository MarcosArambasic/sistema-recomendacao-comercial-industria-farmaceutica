import json,sys
# uso: python _add.py < lote.txt ; linhas: REF|AAAA-MM-DD|fonte|url|titulo|descricao|consulta
out=open('coleta_bruta.jsonl','a',encoding='utf-8')
n=0
for l in sys.stdin.read().splitlines():
    l=l.strip()
    if not l or l.startswith('#'): continue
    p=l.split('|')
    assert len(p)==7,(len(p),l[:80])
    r,dt,f,u,t,d,q=[x.strip() for x in p]
    assert r in('A','M','B'),r
    out.write(json.dumps(dict(ref=r,data=dt,fonte=f,url=u,titulo=t,descricao=d,consulta=q),ensure_ascii=False)+'\n');n+=1
print('ok',n)
