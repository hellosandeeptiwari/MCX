"""Type stubs for kiteconnect.connect — KiteConnect API client."""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class KiteConnect:
    # Exchange constants
    EXCHANGE_NSE: str
    EXCHANGE_BSE: str
    EXCHANGE_NFO: str
    EXCHANGE_CDS: str
    EXCHANGE_BCD: str
    EXCHANGE_BFO: str
    EXCHANGE_MCX: str

    # Transaction types
    TRANSACTION_TYPE_BUY: str
    TRANSACTION_TYPE_SELL: str

    # Product types
    PRODUCT_MIS: str
    PRODUCT_CNC: str
    PRODUCT_NRML: str
    PRODUCT_CO: str

    # Order types
    ORDER_TYPE_MARKET: str
    ORDER_TYPE_LIMIT: str
    ORDER_TYPE_SL: str
    ORDER_TYPE_SLM: str

    # Variety
    VARIETY_REGULAR: str
    VARIETY_AMO: str
    VARIETY_CO: str
    VARIETY_ICEBERG: str
    VARIETY_AUCTION: str

    # Validity
    VALIDITY_DAY: str
    VALIDITY_IOC: str
    VALIDITY_TTL: str

    # Position types
    POSITION_TYPE_DAY: str
    POSITION_TYPE_OVERNIGHT: str

    # Status
    STATUS_COMPLETE: str
    STATUS_CANCELLED: str
    STATUS_REJECTED: str

    # GTT
    GTT_TYPE_SINGLE: str
    GTT_TYPE_OCO: str

    # Margin
    MARGIN_EQUITY: str
    MARGIN_COMMODITY: str

    # Internal attributes
    _routes: Dict[str, str]

    def __init__(
        self,
        api_key: str,
        access_token: Optional[str] = ...,
        root: Optional[str] = ...,
        debug: bool = ...,
        timeout: Optional[int] = ...,
        proxies: Optional[Dict[str, str]] = ...,
        pool: Optional[Any] = ...,
        disable_ssl: bool = ...,
    ) -> None: ...

    def set_access_token(self, access_token: str) -> None: ...
    def login_url(self) -> str: ...
    def generate_session(self, request_token: str, api_secret: str) -> Dict[str, Any]: ...
    def invalidate_access_token(self, access_token: Optional[str] = ...) -> bool: ...
    def renew_access_token(self, refresh_token: str, api_secret: str) -> Dict[str, Any]: ...
    def invalidate_refresh_token(self, refresh_token: str) -> bool: ...
    def set_session_expiry_hook(self, method: Any) -> None: ...

    def margins(self, segment: Optional[str] = ...) -> Dict[str, Any]: ...
    def profile(self) -> Dict[str, Any]: ...

    def place_order(
        self,
        variety: str,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str,
        price: Optional[float] = ...,
        validity: Optional[str] = ...,
        validity_ttl: Optional[int] = ...,
        disclosed_quantity: Optional[int] = ...,
        trigger_price: Optional[float] = ...,
        iceberg_legs: Optional[int] = ...,
        iceberg_quantity: Optional[int] = ...,
        auction_number: Optional[int] = ...,
        tag: Optional[str] = ...,
    ) -> str: ...

    def modify_order(
        self,
        variety: str,
        order_id: str,
        parent_order_id: Optional[str] = ...,
        quantity: Optional[int] = ...,
        price: Optional[float] = ...,
        order_type: Optional[str] = ...,
        trigger_price: Optional[float] = ...,
        validity: Optional[str] = ...,
        disclosed_quantity: Optional[int] = ...,
    ) -> str: ...

    def cancel_order(
        self,
        variety: str,
        order_id: str,
        parent_order_id: Optional[str] = ...,
    ) -> str: ...

    def exit_order(
        self,
        variety: str,
        order_id: str,
        parent_order_id: Optional[str] = ...,
    ) -> str: ...

    def orders(self) -> List[Dict[str, Any]]: ...
    def order_history(self, order_id: str) -> List[Dict[str, Any]]: ...
    def order_trades(self, order_id: str) -> List[Dict[str, Any]]: ...
    def trades(self) -> List[Dict[str, Any]]: ...

    def positions(self) -> Dict[str, List[Dict[str, Any]]]: ...
    def holdings(self) -> List[Dict[str, Any]]: ...
    def convert_position(
        self,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        position_type: str,
        quantity: int,
        old_product: str,
        new_product: str,
    ) -> bool: ...

    def ltp(self, instruments: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]: ...
    def quote(self, instruments: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]: ...
    def ohlc(self, instruments: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]: ...

    def instruments(self, exchange: Optional[str] = ...) -> List[Dict[str, Any]]: ...
    def historical_data(
        self,
        instrument_token: int,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        interval: str,
        continuous: bool = ...,
        oi: bool = ...,
    ) -> List[Dict[str, Any]]: ...

    def trigger_range(
        self,
        transaction_type: str,
        *instruments: str,
    ) -> Dict[str, Any]: ...

    def order_margins(self, params: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
    def basket_order_margins(
        self,
        params: List[Dict[str, Any]],
        consider_positions: bool = ...,
        mode: Optional[str] = ...,
    ) -> Dict[str, Any]: ...
    def get_virtual_contract_note(self, params: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...

    # GTT
    def place_gtt(
        self,
        trigger_type: str,
        tradingsymbol: str,
        exchange: str,
        trigger_values: List[float],
        last_price: float,
        orders: List[Dict[str, Any]],
    ) -> Dict[str, Any]: ...
    def get_gtts(self) -> List[Dict[str, Any]]: ...
    def get_gtt(self, trigger_id: int) -> Dict[str, Any]: ...
    def modify_gtt(
        self,
        trigger_id: int,
        trigger_type: str,
        tradingsymbol: str,
        exchange: str,
        trigger_values: List[float],
        last_price: float,
        orders: List[Dict[str, Any]],
    ) -> int: ...
    def delete_gtt(self, trigger_id: int) -> int: ...

    # Mutual funds
    def mf_orders(self, order_id: Optional[str] = ...) -> Union[Dict[str, Any], List[Dict[str, Any]]]: ...
    def place_mf_order(self, tradingsymbol: str, transaction_type: str, quantity: Optional[int] = ..., amount: Optional[float] = ..., tag: Optional[str] = ...) -> str: ...
    def cancel_mf_order(self, order_id: str) -> str: ...
    def mf_sips(self, sip_id: Optional[str] = ...) -> Union[Dict[str, Any], List[Dict[str, Any]]]: ...
    def place_mf_sip(self, tradingsymbol: str, amount: float, initial_amount: Optional[float] = ..., frequency: str = ..., instalments: int = ..., instalment_day: Optional[int] = ..., tag: Optional[str] = ...) -> Dict[str, Any]: ...
    def modify_mf_sip(self, sip_id: str, amount: Optional[float] = ..., status: Optional[str] = ..., frequency: Optional[str] = ..., instalments: Optional[int] = ..., instalment_day: Optional[int] = ...) -> Dict[str, Any]: ...
    def cancel_mf_sip(self, sip_id: str) -> str: ...
    def mf_holdings(self) -> List[Dict[str, Any]]: ...
    def mf_instruments(self) -> List[Dict[str, Any]]: ...

    def get_auction_instruments(self) -> List[Dict[str, Any]]: ...
