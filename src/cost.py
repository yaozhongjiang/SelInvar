"""Cost accounting. Prices are supplied by the user, never guessed."""

def estimate_api_cost(token_in, token_out, input_price_per_million=None, output_price_per_million=None):
    if input_price_per_million is None or output_price_per_million is None: return None
    if token_in is None or token_out is None: return None
    return token_in/1e6*input_price_per_million + token_out/1e6*output_price_per_million
