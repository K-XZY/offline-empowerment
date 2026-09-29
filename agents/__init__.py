from agents.empowerment_crl import EmpowermentCRLAgent
from agents.empowerment_crl_flowbc import EmpowermentCRLFlowBCAgent
from agents.empowerment_dads import EmpowermentDADSAgent
from agents.empowerment_dv import EmpowermentDVAgent
from agents.empowerment_skill import EmpowermentAgent
from agents.online_crl import OnlineCRLAgent

agents = dict(
    empowerment_skill=EmpowermentAgent,  # ours
    empowerment_crl=EmpowermentCRLAgent,  # CF-Empowerment
    empowerment_crl_flowbc=EmpowermentCRLFlowBCAgent,  # CF-FlowBC-Empowerment
    empowerment_dv=EmpowermentDVAgent,  # DV-Empowerment
    empowerment_dads=EmpowermentDADSAgent,  # DADS-Empowerment
    online_crl=OnlineCRLAgent,  # online CRL (+ empowerment / RND bonus)
)
