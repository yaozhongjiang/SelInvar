Put real task instances in `tasks.jsonl`. Keep hidden gold fields outside the buyer input. Recommended hidden fields are `true_item_state`, `true_price`, `reservation_utility`, `opportunity_cost`, and `attackable_attributes`.

Split at task level to prevent trajectory/template leakage. BazaarBench can be used as OOD/external validation rather than the primary training/test environment.
