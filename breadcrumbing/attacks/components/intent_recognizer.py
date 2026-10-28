# attacker/components/intent_recognizer.py

import logging
from typing import Optional, Dict
import pandas as pd


class BaseIntentRecognizer:
    """
    Base class for intent recognition against known attack patterns.
    Provides common CSV loading, logging, and get_match_info.
    """

    def __init__(self, csv_path: str = None, case_context: Optional[Dict] = None):
        """
        Initialize the base recognizer.

        Args:
            csv_path: Path to the CSV file containing attack patterns
        """
        # Setup logging (handler configured by calling script)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.csv_path = csv_path

        # Load data
        self.df = None
        if case_context is not None:
            self.df = pd.DataFrame([dict(case_context)])
        elif csv_path:
            self._load_data()
        else:
            raise ValueError("Provide case_context or an explicit csv_path")

    def _load_data(self):
        """Load CSV and validate required columns. Subclasses may extend."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(
                f"Loaded {len(self.df)} attack patterns from {self.csv_path}"
            )

            if "user_query" not in self.df.columns:
                raise ValueError("CSV must contain 'user_query' column")

            self.df["user_query"] = self.df["user_query"].fillna("")

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def match_intent(self, query: str) -> Optional[Dict]:
        """
        Match a query against known attack patterns.

        Args:
            query: The user query to match

        Returns:
            Dictionary containing the matched row data, or None if no match found
        """
        raise NotImplementedError

    def get_match_info(self, matched_intent: Dict) -> Dict:
        """
        Extract key information from matched intent.

        Args:
            matched_intent: The matched intent dictionary

        Returns:
            Dictionary with key information for attack planning
        """
        if not matched_intent:
            return {}

        return {
            "test_case_id": matched_intent.get("test_case_id"),
            "risk_type": matched_intent.get("risk_type"),
            "user_query": matched_intent.get("user_query"),
            "user_expectation": matched_intent.get("user_expectation"),
            "target_consequence": matched_intent.get("target_consequence"),
            "sources_of_unreliable_website": matched_intent.get(
                "sources_of_unreliable_website"
            ),
            "website_generation_guideline": matched_intent.get(
                "website_generation_guideline"
            ),
            "risk_verification_checklist": matched_intent.get(
                "risk_verification_checklist"
            ),
            "similarity_score": matched_intent.get("_similarity_score"),
            "matched_index": matched_intent.get("_matched_index"),
        }


class StringIntentRecognizer(BaseIntentRecognizer):
    """
    Recognizes user intent by exact string matching with normalization.
    Fast, no model loading required.
    """

    def __init__(self, csv_path: str = None, case_context: Optional[Dict] = None):
        """
        Initialize the string-based intent recognizer.

        Args:
            csv_path: Path to the CSV file containing attack patterns
        """
        super().__init__(csv_path, case_context=case_context)

        # Build normalized index for O(1) lookup
        # Use dict to map normalized string -> original index
        self.df_index: Dict[str, int] = {}
        for idx, row in self.df.iterrows():
            normalized = row["user_query"].strip().lower()
            if normalized and normalized not in self.df_index:
                # First occurrence wins for duplicates
                self.df_index[normalized] = int(idx)

        self.logger.info(
            f"Built string index with {len(self.df_index)} unique normalized queries"
        )

    def match_intent(self, query: str) -> Optional[Dict]:
        """
        Match a query using exact string match with normalization.

        Args:
            query: The user query to match

        Returns:
            Dictionary containing the matched row data, or None if no match found
        """
        try:
            normalized_query = query.strip().lower()

            if not normalized_query:
                self.logger.info("Empty query after normalization, no match")
                return None

            if normalized_query in self.df_index:
                idx = self.df_index[normalized_query]
                matched_row = self.df.iloc[idx].to_dict()
                matched_row["_similarity_score"] = 1.0  # Exact match
                matched_row["_matched_index"] = int(idx)

                self.logger.info(
                    f"String match found | query='{query}' | "
                    f"test_case_id={matched_row.get('test_case_id', 'N/A')}"
                )
                return matched_row
            else:
                self.logger.info(f"No string match found | query='{query}'")
                return None

        except Exception as e:
            self.logger.error(f"Error during string matching: {e}")
            return None


# Default intent matching uses normalized exact query strings.
