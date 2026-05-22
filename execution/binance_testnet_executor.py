"""
ATLAS Binance Spot Testnet Executor

Executes paper-safe orders against Binance Spot Testnet.
Requires:
- BINANCE_TESTNET_API_KEY
- BINANCE_TESTNET_API_SECRET
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict
from urllib.parse import urlencode

import httpx


class BinanceTestnetExecutor:
    BASE_URL = "https://testnet.binance.vision"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        import os
        self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise RuntimeError("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET are required")

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "LIMIT",
        price: float | None = None,
        time_in_force: str = "GTC",
        test_only: bool = False,
    ) -> Dict[str, Any]:
        endpoint = "/api/v3/order/test" if test_only else "/api/v3/order"

        params: Dict[str, Any] = {
            "symbol": symbol.upper().replace("/", ""),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": qty,
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
        }

        if params["type"] == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = price
            params["timeInForce"] = time_in_force

        params["signature"] = self._sign(params)

        headers = {"X-MBX-APIKEY": self.api_key}

        async with httpx.AsyncClient(base_url=self.BASE_URL, timeout=20.0) as client:
            response = await client.post(endpoint, params=params, headers=headers)

        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}

        return {
            "broker": "binance_testnet",
            "endpoint": endpoint,
            "status_code": response.status_code,
            "accepted": response.status_code in (200, 201),
            "symbol": params["symbol"],
            "side": params["side"],
            "type": params["type"],
            "quantity": qty,
            "price": price,
            "response": payload,
        }
