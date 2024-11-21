# get streaming price data from oanda broker
# Oanda streaming api
import config.acct_config as ac 
import v20

# Set up your OANDA account details
account_id = ac.ACCOUNT_ID
access_token = ac.API_KEY

# Create the API context
api = v20.Context(
    hostname="stream-fxtrade.oanda.com", # or "stream-fxpractice.oanda.com"
    port="443",
    token=access_token,
)

# Define instruments you want to stream (e.g., EUR/USD, USD/JPY)
instruments = "EUR_USD,USD_JPY"

# Stream real-time pricing and print heartbeats/errors
response = api.pricing.stream(
    account_id,
    instruments=instruments
)

# Listen for incoming messages and handle different types
def stream_prices():
    for msg_type, msg in response.parts():
        if isinstance(msg, v20.pricing.ClientPrice):
            print(f"Instrument: {msg.instrument}, Time: {msg.time}, Status: {msg.status}, Tradeable: {msg.tradeable}")
            print("Bids:")
            for bid in msg.bids:
                print(f"  Price: {bid.price}, Liquidity: {bid.liquidity}")
            print("Asks:")
            for ask in msg.asks:
                print(f"  Price: {ask.price}, Liquidity: {ask.liquidity}")
            print(f"Closeout Bid: {msg.closeoutBid}, Closeout Ask: {msg.closeoutAsk}")
            print("-" * 40)  # Separator for readability
        elif isinstance(msg, v20.pricing.PricingHeartbeat):
            print(f"Heartbeat received at: {msg.time}")
        else:
            print(f"Unexpected message type: {msg_type}, message: {msg}")


# get historical data






# stream_prices()