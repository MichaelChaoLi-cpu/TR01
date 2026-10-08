"""Acquire two existing-case manager plans only, with preserved receipts."""
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-crosswalk/boundary-evidence-closeout'
PAGES = {
    'S03': 'https://www.city.hasuda.saitama.jp/doro/documents/kyouryou.html',
    'S12': 'https://www.city.ushiku.lg.jp/page/page008311.html',
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.current=None
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.current=[dict(attrs).get('href',''),'']
    def handle_data(self,data):
        if self.current is not None: self.current[1]+=data
    def handle_endtag(self,tag):
        if tag=='a' and self.current is not None:
            self.links.append(self.current); self.current=None


def save(url,name):
    path=OUT/'sources'/name; receipt_path=path.with_suffix(path.suffix+'.request.json')
    if path.exists() and receipt_path.exists():
        data=path.read_bytes(); receipt=json.loads(receipt_path.read_text())
        assert receipt['url']==url and hashlib.sha256(data).hexdigest()==receipt['sha256']
    else:
        with urllib.request.urlopen(url,timeout=40) as response:
            data=response.read(12*1024*1024+1); final_url=response.url
        assert len(data)<=12*1024*1024
        receipt={'url':url,'resolved_url':final_url,'bytes':len(data),
                 'sha256':hashlib.sha256(data).hexdigest(),
                 'retrieved_utc':datetime.now(timezone.utc).isoformat()}
        path.write_bytes(data);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'file':name,**receipt}),flush=True)
    return data,{'file':name,**receipt}


def main():
    (OUT/'sources').mkdir(parents=True,exist_ok=True)
    cases=json.loads((OUT.parent/'stratified-review/review_cases.json').read_text())
    manifest=[]; originals=[]
    for id,url in PAGES.items():
        case=next(c for c in cases if c['review_id']==id)
        source=ROOT/'data/exp/feasibility-check/national-bridge-inventory'/case['source_page']
        record=next(r for r in json.loads(gzip.decompress(source.read_bytes()))['result'] if r['shisetsu_id']==case['shisetsu_id'])
        expected='蓮田市' if id=='S03' else '牛久市'
        assert record['syogen']['kanrisya']['meisyou']==expected
        originals.append({'review_id':id,'source_page':case['source_page'],
                          'source_page_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                          'record':record})
        html,receipt=save(url,id+'_manager_page.html');manifest.append(receipt)
        parser=Links();parser.feed(html.decode('utf-8'))
        links=[(urllib.parse.urljoin(url,href),text.strip()) for href,text in parser.links if '.pdf' in href.lower() and '長寿命化' in text]
        discovery='manager_page_link'
        if id=='S12' and not links:
            # Official-domain PDF observed independently in the web source lookup.
            # The saved parent response is an HTML meta-refresh redirect, not the plan.
            pdf_url='https://www.city.ushiku.lg.jp/data/doc/1736989475_doc_352_0.pdf'
            title='牛久市橋梁長寿命化修繕計画'
            discovery='official_domain_pdf_from_web_lookup; manager_page_returned_meta_refresh_redirect'
        else:
            assert len(links)==1, f'Inspect the parent page before choosing among document links: {links}'
            pdf_url,title=links[0]
        time.sleep(1)
        data,receipt=save(pdf_url,id+'_manager_plan.pdf')
        assert data.startswith(b'%PDF')
        manifest.append({'manager_page_url':url,'document_title':title,'discovery_method':discovery,**receipt})
        time.sleep(1)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'source_asset_records.json').write_text(json.dumps(originals,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
