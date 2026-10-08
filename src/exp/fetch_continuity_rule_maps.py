"""Reuse GSI evidence rendering for the independent rule sample."""
import fetch_national_quality_maps as maps
from validate_national_continuity_rule import OUT
if __name__=='__main__':
    maps.OUT=OUT
    maps.main()
