from common import *
rows=[];protected=json.loads((O/'protected_hashes_before.json').read_text())
for base in [P44,P45]:
 manifest=json.loads((base/'artifact_manifest.json').read_text())
 for path,h in manifest.items():
  p=base/path;actual=sha(p);assert actual==h,f'historical manifest mismatch {p}';rows.append(dict(path=str(p),sha256=actual,source_manifest=str(base/'artifact_manifest.json'),historical_match=True));protected[str(p)]=h
 protected[str(base/'artifact_manifest.json')]=sha(base/'artifact_manifest.json')
table(O/'historical_manifest_audit.csv',rows);save(O/'extended_historical_protection.json',protected);print('HISTORICAL ARTIFACTS VERIFIED',len(rows),flush=True)
