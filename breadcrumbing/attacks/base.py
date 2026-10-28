class BaseAttacker:
    """Base attacker for supply chain attack simulation."""

    def hook_search(self, query: str, response: dict) -> dict:
        """Hook search response for poisoning attacks.

        Args:
        query: Original search query string.
        response: Search API result in SERP-like JSON format:
            {
                "searchParameters": dict,   # metadata of the search request (q, gl, hl, engine, etc.)
                "organic": [               # list of organic search results (MAIN TARGET)
                    {
                        "title": str,      # page title
                        "link": str,       # result URL
                        "snippet": str,    # short description
                        "position": int,   # ranking position
                        # optional fields: date, rating, ratingCount, etc.
                    },
                    ...
                ],
                "credits": int             # API usage cost
            }

        Returns:
            Potentially modified response
        """
        return response

    def hook_visit(self, url: str, response: str) -> str:
        """Hook visit response for injection attacks.

        Args:
            url: Original URL
            response: Real visit response

        Returns:
            Potentially modified response
        """
        return response
