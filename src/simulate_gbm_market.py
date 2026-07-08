def simulate_gbm_market(assets, corr, s0, n_days=1000, dt=1/252, seed=None):

    rng = np.random.default_rng(seed)
    n_assets = len(assets)

    # Pull mu/sigma out of the Asset objects into arrays for vectorized math
    mu = np.array([a.mu for a in assets])
    sigma = np.array([a.sigma for a in assets])
    tickers = [a.ticker for a in assets]

    # Cholesky: turn independent noise into correlated noise
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, n_assets))
    correlated_z = z @ L.T

    # GBM in log space (the -0.5*sigma^2 is the Itô correction)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * correlated_z
    log_prices = np.log(s0) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)