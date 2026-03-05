# -*- coding: utf-8 -*-
"""
Dev script to clean LLM metadata JSON (scraped via Perplexity).
Should be incorporated into a dta research pipeline later (post-MVP).
"""

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


if __name__ == "__main__":
    project_root = str(Path(Path(__file__).resolve().parents[1], "src"))
    if project_root not in sys.path:
        sys.path.append(project_root)

from llm_compass.data.models import LLMMetadataSchema
from llm_compass.common.types import Modality, MODALITY_VALUES

SOURCE_LLM_METADATA = Path(Path(__file__).parent / "llm_metadata_scraped.json")
TARGET_LLM_METADATA = Path(Path(__file__).parent / "llm_metadata_cleaned.json")

SOURCE_PROVIDERS = Path(Path(__file__).parent / "provider_metadata.csv")
PROVIDERS = pd.read_csv(SOURCE_PROVIDERS)


def cost_translate(provider: str, modality: str, mode: str) -> float | None:
    """Factor by which image, audio or video translate into number of tokens,
    i.e. 1 image, 1 hour of audio or 1 second of video are how many tokens?

    These factors were manually researched in provider_metadata.csv

    Assumptions:
    - image: 1024 x 1024 pixels and average / medium / standard quality
    - audio: input as preprocessed by the model, output 44.1 kHz mono
    - video: HD (1280x720) at 30fps
    """
    if provider not in PROVIDERS.provider.values:
        raise ValueError(f"❌ Provider not found: {provider}")
    if modality not in MODALITY_VALUES:
        raise ValueError(f"❌Invalid modality: {modality}")
    if mode not in ("input", "output"):
        raise ValueError(f"❌Invalid mode: {mode}")
    if modality == "text":
        return 1.0

    key_col = f"tokens_{modality}_{mode}"
    unit_col = f"tokens_{modality}_{mode}_unit"
    value = PROVIDERS.loc[PROVIDERS.provider == provider, key_col].iat[0]
    if pd.isna(value):
        return None

    value = float(value)  # type: ignore

    if modality == "image":
        return value

    unit = PROVIDERS.loc[PROVIDERS.provider == provider, unit_col].iat[0]
    if modality == "audio" and unit == "second":
        return value * 3600  # per hour
    elif modality == "audio" and unit == "minute":
        return value * 60
    elif modality == "audio" and unit == "hour":
        return value
    elif modality == "video" and unit == "second":
        return value  # per second
    elif modality == "video" and unit == "minute":
        return value / 60
    elif modality == "video" and unit == "hour":
        return value / 3600
    else:
        raise ValueError(f"Invalid unit in providers dataframe for {provider}: {unit}")


class LLMMetadataManager:
    int_cols = [
        k
        for k, v in LLMMetadataSchema.model_fields.items()
        if v.annotation in (int, Optional[int])
    ]
    float_cols = [
        k
        for k, v in LLMMetadataSchema.model_fields.items()
        if v.annotation in (float, Optional[float])
    ]

    def __init__(self, source=SOURCE_LLM_METADATA):
        self.source = source
        with open(self.source, "r") as f:
            self.llms = json.load(f)
        self.header = self.llms[0].keys() if self.llms else []

    def all_names(self) -> list[str]:
        names = sorted([llm["name_normalized"] for llm in self.llms])
        out = []
        for name in names:
            out.append(name)
            llm = next(
                (llm for llm in self.llms if llm["name_normalized"] == name),
                None,
            )
            if llm and llm["name_aliases"]:
                names_alt = [_.strip() for _ in llm["name_aliases"].split(",")]
                out.extend(names_alt)
        return sorted(out)

    def all_providers(self):
        providers = sorted(list(set([llm["provider"] for llm in self.llms])))
        return providers

    def check_provider_coverage(self):
        providers = self.all_providers()
        missing = [p for p in providers if p not in PROVIDERS.provider.values]
        if missing:
            print("❌ Providers missing from provider_metadata.csv:")
            for p in missing:
                print(f"    {p}")
        else:
            print("✅ All providers are mentioned in provider_metadata.csv")

    def check_duplicate_names(self):
        names = self.all_names()
        duplicates = set([x for x in names if names.count(x) > 1])
        if duplicates:
            print("❌ Duplicate names found:")
            for name in duplicates:
                print(name)
        else:
            print("✅ No duplicate names found.")

    def convert_strings(self):
        """JSON may contain int and float encoded as strings.
        This converts them to the correct types.
        """
        for llm in self.llms:
            # print(f"Processing {llm['name_normalized']}")
            for key, value in llm.items():
                if key in self.int_cols and isinstance(value, str):
                    if value == "":
                        llm[key] = None
                        continue
                    try:
                        llm[key] = int(value)
                    except ValueError:
                        print(
                            f"❌ {llm['name_normalized']}: Expected int for {key} but got '{value}'"
                        )
                elif key in self.float_cols and isinstance(value, str):
                    if value == "":
                        llm[key] = None
                        continue
                    try:
                        llm[key] = float(value)
                    except ValueError:
                        print(
                            f"❌ {llm['name_normalized']}: Expected float for {key} but got '{value}'"
                        )

    def check_cost_fields(self):
        """Checks that cost fields are present for each modality.

        Be sure to call .convert_strings() first
        """
        suffix = {
            "text": "_1m",
            "image": "_1024",
            "audio": "_1h",
            "video": "_1s",
        }
        for llm in self.llms:
            for mode in ("input", "output"):
                for modality in llm[f"modality_{mode}"].split(","):
                    modality = modality.strip()
                    cost_key = f"cost_{mode}_{modality}{suffix[modality]}"
                    cost_key_text = f"cost_{mode}_text_1m"
                    if llm[cost_key] in ("", None):
                        # try to fill
                        ct = cost_translate(llm["provider"], modality, mode)
                        if ct is not None and llm[cost_key_text] not in ("", None):
                            llm[cost_key] = ct * llm[cost_key_text] / 1e6
                        elif ct is None:
                            print(
                                f"⚠️ {llm['name_normalized']:<30s}: Missing cost field {cost_key}."
                            )
                        else:
                            print(
                                f"⚠️ {llm['name_normalized']:<30s}: Missing cost field {cost_key}. "
                                f"Translates to {ct} text tokens."
                            )

    def write_cleaned(self, target=TARGET_LLM_METADATA):
        with open(target, "w") as f:
            json.dump(self.llms, f, indent=2)
        print(f"✅ Cleaned LLM metadata written to {target}")


if __name__ == "__main__":
    manager = LLMMetadataManager()
    manager.check_duplicate_names()
    manager.check_provider_coverage()
    manager.convert_strings()
    manager.check_cost_fields()
    manager.write_cleaned()
