from pydantic import BaseModel


class NormalizedName(BaseModel):
    raw: str
    canonical: str
    tokens: list[str]
    first: str
    middle: list[str]
    last: str
    has_initials: bool  # at least one single-letter token (e.g. "J. Adams")
    reordered: bool  # "Last, First" was converted to "first last"


class NormalizedAddress(BaseModel):
    raw: str
    canonical: str
    parsed: bool  # False when the address did not match the expected structure
    number: str | None = None
    street: str | None = None
    city: str | None = None
    country: str | None = None
