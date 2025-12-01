# QuantraCore Apex — Risk Disclosure Statement

**Version:** 1.0  
**Effective Date:** 2025-01-01  
**Last Updated:** 2025-12-01

---

## Important Notice

PLEASE READ THIS RISK DISCLOSURE STATEMENT CAREFULLY BEFORE USING QUANTRACORE APEX. BY USING THE PLATFORM, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND ACCEPT THESE RISKS.

---

## 1. General Trading Risks

### 1.1 Market Risk
Trading in securities involves substantial risk of loss. The value of investments can fluctuate significantly due to:
- Economic conditions and market sentiment
- Company-specific developments
- Sector and industry dynamics
- Geopolitical events
- Interest rate changes
- Currency fluctuations

### 1.2 Volatility Risk
Securities prices can be highly volatile. Small-cap, penny stocks, and momentum plays—which are focus areas of this Platform—are particularly volatile:
- Penny stocks (sub-$5) can move 20-50%+ in a single day
- Low-float stocks can experience extreme price swings
- Volume surges can lead to slippage and poor execution

### 1.3 Liquidity Risk
Some securities may have limited liquidity:
- Wide bid-ask spreads can increase transaction costs
- Large orders may not be filled at expected prices
- Exit during market stress may be difficult or impossible

### 1.4 Leverage Risk
The Platform supports margin trading up to 4x leverage:
- Losses are magnified proportionally to leverage used
- Margin calls may force liquidation at unfavorable prices
- Interest costs reduce overall returns

---

## 2. AI/ML Model Risks

### 2.1 Model Accuracy
The ApexCore machine learning models, while trained on real market data, have inherent limitations:

| Model Head | Accuracy | What This Means |
|------------|----------|-----------------|
| Runner Probability | 98.88% | ~1% of runner predictions are incorrect |
| Quality Score | 93.75% | ~6% of quality assessments may be wrong |
| Avoid Probability | 99.91% | Very rare false positives for avoid signals |
| Timing Prediction | 94.4% | ~6% of timing buckets may be incorrect |
| Regime Detection | 95.25% | ~5% of market regime classifications may be wrong |

### 2.2 Model Drift
Model performance may degrade over time due to:
- Changes in market structure
- Regime shifts not in training data
- New market participants and strategies
- Regulatory changes

### 2.3 Training Data Limitations
Models are trained on historical data that may not reflect future conditions:
- Training period: 2 years of 15-minute intraday bars
- Training symbols: 59 symbols (may not generalize to all stocks)
- Training conditions: May not include rare events (flash crashes, pandemics)

### 2.4 Overfitting Risk
Despite validation procedures, models may be overfit to historical patterns that don't persist.

---

## 3. Platform-Specific Risks

### 3.1 Technical Risks
- System outages may prevent signal generation or trade execution
- Network latency may cause delayed signals
- Software bugs may produce incorrect predictions
- API failures from third-party providers (Alpaca, Polygon)

### 3.2 Data Feed Risks
The Platform relies on external data sources:
- IEX data feed (via Alpaca) may have delayed or missing data
- Polygon data may have rate limits affecting real-time updates
- Data errors can propagate to incorrect predictions

### 3.3 Execution Risks
For signals used in manual trading:
- Market conditions may change between signal and execution
- Your broker's execution quality may vary
- Slippage may differ from predicted entry/exit levels

---

## 4. Strategy-Specific Risks

### 4.1 Momentum Trading
- "Runner" strategies depend on continuation of price momentum
- Reversals can occur suddenly without warning
- High conviction signals can still result in losses

### 4.2 Penny Stock Trading
- Lower regulatory oversight and disclosure requirements
- Higher susceptibility to manipulation
- Greater potential for total loss of investment

### 4.3 Short Selling
- Theoretically unlimited loss potential
- Short squeezes can cause rapid, extreme losses
- Borrowing costs and availability may vary

### 4.4 Intraday Trading
- Pattern day trader rules apply to accounts under $25,000
- Transaction costs accumulate with frequency
- Emotional fatigue can impair decision-making

---

## 5. Regulatory and Legal Risks

### 5.1 Regulatory Changes
- Securities regulations may change affecting Platform operations
- Tax treatment of trading gains may be modified
- Broker regulatory requirements may change

### 5.2 Broker Risk
- Alpaca Markets is a registered broker-dealer
- Accounts are protected by SIPC (up to $500,000 securities)
- Broker failure could affect access to funds

---

## 6. Risk Management Measures

The Platform includes risk management features, but these do not eliminate risk:

| Control | Description | Limitation |
|---------|-------------|------------|
| Position Limits | Max $10K per symbol | Does not prevent total portfolio loss |
| Exposure Limits | Max $100K total | Does not guarantee capital preservation |
| Stop-Loss | ATR-based protective stops | Gaps can skip stop levels |
| Omega Directives | Safety override protocols | May miss novel risk scenarios |
| Risk Tiers | Conservative to aggressive scaling | User can override recommendations |

---

## 7. Acknowledgment

By using QuantraCore Apex, you acknowledge and accept:

1. **I understand that trading involves substantial risk of loss**, including potential loss of my entire investment.

2. **I understand that past performance does not guarantee future results.** Historical accuracy metrics do not predict future model performance.

3. **I understand that the Platform provides signals and analysis, not investment advice.** All trading decisions are my own responsibility.

4. **I have the financial capacity to bear potential losses** from trading activities.

5. **I will not trade with money I cannot afford to lose.**

6. **I am solely responsible for my trading decisions** and their outcomes.

7. **I have read and understood this Risk Disclosure Statement** in its entirety.

---

## 8. Additional Resources

- [FINRA Investor Education](https://www.finra.org/investors)
- [SEC Investor Alerts](https://www.sec.gov/investor-alerts)
- [Alpaca Risk Disclosures](https://alpaca.markets/disclosures)

---

**Document Checksum:** SHA-256 will be generated on finalization  
**Legal Review:** Pending
