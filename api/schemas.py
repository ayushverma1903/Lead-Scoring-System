from pydantic import BaseModel
from typing import List, Dict, Any

class RawLead(BaseModel):
    """
    A single lead's raw data. Accepts any of the original dataset's
    columns as key-value pairs (e.g. 'TotalVisits', 'Lead Source', etc.)
    Missing fields are handled automatically by the preprocessing pipeline.
    """
    data: Dict[str, Any]

class RawLeadsBatch(BaseModel):
    """
    A batch of leads for scoring multiple instances at once.
    """
    leads: List[Dict[str, Any]]
