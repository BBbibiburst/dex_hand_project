from pathlib import Path
import json,hashlib,shutil,xml.etree.ElementTree as ET
R=Path.cwd();O=R/'docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=R/'thesis/body/ch02_simulation_platform.tex';shutil.copyfile(p,O/'chapter_before.tex')
paths=[q for q in (R/'thesis').rglob('*') if q.is_file() and '.tools' not in q.parts and ('releases' in q.parts or q.suffix in ['.tex','.bib','.cls','.cfg','.sty','.bst'])]
(O/'protected_before.json').write_text(json.dumps({str(q.relative_to(R)):sha(q) for q in paths},indent=2)+'\n')
rows=[]
for folder in ['assets/robots/rm75b/rm75b.xml','assets/grippers/dex_hand/dex_hand.xml']:
 root=ET.parse(folder).getroot()
 for a in root.find('actuator'):
  rows.append({'source':folder,'type':a.tag,**a.attrib})
(O/'actuator_xml_check.json').write_text(json.dumps({'actuators':rows,'fixed_tendons':[{'name':a.get('name'),'terms':[x.attrib for x in a]} for a in ET.parse('assets/grippers/dex_hand/dex_hand.xml').getroot().find('tendon')],'solver_override_source':'source/robots/builder.py::_configure_solver','solver_overrides_actuator_parameters':False,'units':'SI model convention; not calibrated hardware values; arm kp Nm/rad; tendon kp N/m'},indent=2)+'\n')
