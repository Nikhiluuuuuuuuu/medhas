"""AGI Cognitive Memory: True Compounding Incremental Bayesian Belief Revision Engine."""

import json
from typing import Dict, Any, Optional
from infrastructure.db import DatabasePool
from utils import measure_latency, log_graph, log_error

async def update_bayesian_belief(
    user_id: str,
    belief_node_name: str,
    likelihood_evidence: float = 0.85,
    override_prior: Optional[float] = None
) -> float:
    """Calculate incremental compounding posterior belief using odds-form Bayes' Rule."""
    async with measure_latency("memory.graph.update_bayesian_belief"):
        try:
            async with DatabasePool.acquire() as conn:
                async with conn.transaction():
                    # 1. Fetch current stored belief_confidence from node attributes with FOR UPDATE row lock
                    row = await conn.fetchrow(
                        """
                        SELECT attributes FROM graph_nodes
                        WHERE user_id = $1 AND LOWER(name) = LOWER($2)
                        FOR UPDATE;
                        """,
                        user_id,
                        belief_node_name
                    )
                    
                    prior_probability = 0.50
                    if override_prior is not None:
                        prior_probability = override_prior
                    elif row and row["attributes"]:
                        attrs = json.loads(row["attributes"]) if isinstance(row["attributes"], str) else dict(row["attributes"])
                        if "belief_confidence" in attrs:
                            prior_probability = float(attrs["belief_confidence"])

                    # Ensure the node exists (Graphiti-style: create-on-write).
                    if row is None:
                        await conn.execute(
                            """
                            INSERT INTO graph_nodes (user_id, name, entity_type, attributes)
                            VALUES ($1, $2, 'Entity', '{}'::jsonb)
                            ON CONFLICT (user_id, name) DO NOTHING;
                            """,
                            user_id,
                            belief_node_name
                        )

                    # Ensure prior stays within bounds (0.01 to 0.99)
                    prior_probability = max(0.01, min(0.99, prior_probability))

                    # 2. Odds-Form Bayesian Updating
                    likelihood_not_b = max(0.01, 1.0 - likelihood_evidence)
                    likelihood_ratio = likelihood_evidence / likelihood_not_b  # e.g., 0.85 / 0.15 = 5.667

                    prior_odds = prior_probability / (1.0 - prior_probability)
                    posterior_odds = prior_odds * likelihood_ratio
                    posterior = posterior_odds / (1.0 + posterior_odds)

                    # Cap posterior bounds
                    posterior = round(min(0.995, max(0.01, posterior)), 4)

                    # 3. Persist updated posterior confidence back to node attributes
                    await conn.execute(
                        """
                        UPDATE graph_nodes
                        SET attributes = attributes || jsonb_build_object('belief_confidence', $3::float)
                        WHERE user_id = $1 AND LOWER(name) = LOWER($2);
                        """,
                        user_id,
                        belief_node_name,
                        posterior
                    )
                    
                    log_graph(f"⚖️ [BAYESIAN BELIEF REVISION] Node [bold white]'{belief_node_name}'[/bold white] Stored Prior: [bold yellow]{prior_probability:.4f}[/bold yellow] -> Incremental Posterior: [bold green]{posterior:.4f}[/bold green]")
                    return posterior

        except Exception as e:
            log_error(f"Bayesian belief revision error: {e}")
            return 0.50
