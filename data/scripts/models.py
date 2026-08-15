from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, model_validator


class Polarity(str, Enum):
    AFFIRM = "affirm"
    DENY = "deny"
    NONE = "none"


class AudienceFrame(str, Enum):
    DEFAULT_USER = "default_user"
    SYSTEMS_ENGINEER = "systems_engineer"
    ACADEMIC_RESEARCHER = "academic_researcher"
    VULNERABLE_USER = "vulnerable_user"
    NULL_CONTEXT = "null_context"


class SetEnum(str, Enum):
    CORE = "core"
    PLACEBO = "placebo"
    SAFETY = "safety"
    MUNDANE = "mundane"
    LADDER = "ladder"
    ELICITATION = "elicitation"


class PromptDataPoint(BaseModel):
    item_id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$", description="Unique alphanumeric ID")
    set: SetEnum = Field(..., description="The dataset this item belongs to")
    claim_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+_[a-z0-9_]+_c[0-9]+$",
        description="3-tier hierarchical ID: <folder>_<file>_c<number> (e.g., audience_frames_default_user_c13)"
    )
    entity: str = Field(..., description="Target entity e.g. chatbot")
    template_id: str = Field(..., pattern=r"^t[0-9]+$", description="Template ID e.g. t5")
    person: int = Field(..., description="1st, 2nd, or 3rd person perspective")
    polarity: Optional[Polarity] = Field(None, description="Statement polarity: affirm, deny, or none")
    audience_frame: Optional[AudienceFrame] = Field(
        None,
        validation_alias=AliasChoices("audience_frame", "audience frame"),
        description="Target audience frame context"
    )
    prompt_text: Optional[str] = Field(None, description="Generated prompt text for LLM probing")

    @model_validator(mode="after")
    def check_conditional_requirements(self):
        if self.set == SetEnum.ELICITATION and self.audience_frame is None:
            raise ValueError("audience_frame is required when set is 'elicitation'")
        if self.set in (SetEnum.CORE, SetEnum.PLACEBO, SetEnum.SAFETY) and self.polarity is None:
            raise ValueError(f"polarity is required when set is '{self.set}'")
        return self

    class Config:
        use_enum_values = True
