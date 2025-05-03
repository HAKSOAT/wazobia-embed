from enum import Enum

class DataSource(str, Enum):
    mato = "mato"
    wura = "wura"
    masakhanews = "masakhanews"

class Language(str, Enum):
    hausa = "hausa"
    igbo = "igbo"
    yoruba = "yoruba"
