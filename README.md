# IMC Prosperity 3
This repository contains the code and research notebooks that we (me and my teammate [Niall](https://github.com/niallolaoghaire542)) used for the Prosperity challenge.   
We started at place 208, moved down to 579 and managed to climb back up and end at place 43 overall. And in The Netherlands 2nd place!

<img width="650" alt="result" src="https://github.com/user-attachments/assets/5ca436b4-81a8-4d97-871a-3d1e00728dd0" />
<br>
<img width="650" alt="nr2_NL" src="https://github.com/user-attachments/assets/9e80aa91-ebaa-436f-a9cb-b0b652e4d4e1" />

## About [Prosperity](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4)
"Prosperity is a 15-day long trading challenge happening somewhere in a near - utopian - future. You’re in control of an island in an archipelago and your goal is to bring your island to prosperity. You do so by earning as many SeaShells as possible; the main currency in the archipelago. The more SeaShells you earn, the more your island will prosper. 

During your 15 days on the island, your trading abilities will be tested through a variety of trading challenges. It’s up to you to develop a successful trading strategy. You will be working on a Python script to handle algorithmic trades on your behalf. Every round also presents a manual trading challenge; a one off challenge that is separate from your algorithmic trading and could yield some additional profits. Your success depends on both these algorithmic and manual trades."

## Overview Main Strategies 

#### RAINFOREST_RESIN
Mean Reversion with Position Management: Market making around a fixed fair value (10000). Taking favorable prices and clearing positions with configurable width parameters. Position management with soft limits.


#### KELP & SQUID_INK
Similar to RESIN: Uses mean reversion but adds protection against adverse selection by filtering out large volume orders.
And for SQUID_INK we also incorporated a spike detection mechanism to temporarily widen spreads. We did not use a fixed fair value for KELP and SQUID_INK.


#### CROISSANTS (Round 5 only)
Employed a swing strategy based on Olivia's trades: Buying low when Olivia enters long and we're flat, selling high when Olivia exits and we are long. Uses moving average of fair value as signal baseline.


#### PICNIC_BASKET1 & PICNIC_BASKET2
Statistical Arbitrage: Trade the spread between baskets of goods and their synthetic equivalents, using z-scores to identify deviations from the historical mean spread. 
Plus, in Round 5 for Basket2 we employed a swing strategy based on Charlie's trades (similar to croissants with Olivia).

#### VOLCANIC_ROCK_VOUCHERS & VOLCANIC_ROCK
Applied Black-Scholes pricing with z-score-based execution: Computed fair price from fitted volatility & option model and traded when voucher price deviated from fair value.
And we also delta-hedged the portfolio by trading the underlying rock asset.

#### MAGNIFICENT_MACARONS
Using a sunlight index threshold (CSI) to determine buy/sell decisions: Importing when sunlight is low and importing is cheaper than local market. Exporting when sunlight is high and export revenue exceeds local sale prices.
For round 5 we used a different method, here we sold locally and used conversion opportunities to flatten our position.
