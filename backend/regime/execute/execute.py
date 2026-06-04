import time
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.insert(0, '/home/vmc01/vinh/noraquantengine/Grey/backend')

from core.position_sizer import ExecutionPayload, OrderTranche


class ExecutionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class ExecutionReport:
    status: ExecutionStatus
    executed_orders: List[Dict] = field(default_factory=list)
    failed_tranches: List[int] = field(default_factory=list)
    
    total_filled_lots: float = 0.0
    total_rejected_lots: float = 0.0
    
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    execution_time_ms: float = 0.0
    average_slippage_pct: float = 0.0


@dataclass
class ExecutionConfig:
    max_spread_pct: float = 0.002
    max_slippage_pct: float = 0.001
    
    order_retry_attempts: int = 3
    retry_delay_ms: int = 500
    
    market_order_timeout_sec: int = 5
    limit_order_timeout_sec: int = 2
    
    enable_emergency_cancel: bool = True
    min_liquidity_lots: float = 10.0
    
    rate_limit_delay_ms: int = 100


class ExecutionCommander:
    
    def __init__(self, 
                 exchange_api,
                 config: Optional[ExecutionConfig] = None,
                 enable_live_checks: bool = True,
                 dry_run: bool = False):
        
        self.api = exchange_api
        self.config = config or ExecutionConfig()
        self.enable_live_checks = enable_live_checks
        self.dry_run = dry_run
        
        self.logger = logging.getLogger("ExecutionCommander")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    
    def verify_state_sync(self,
                         payload: ExecutionPayload,
                         current_ticker: Dict,
                         drift_threshold_atr: float = 0.2,
                         time_threshold_sec: float = 5.0) -> Tuple[bool, Optional[str]]:
        """
        STATE SYNCHRONIZATION VERIFICATION:
        Kiểm tra xem môi trương thị trường có thay đổi quá nhiều trong lúc Regime tính toán không.
        
        Đây là BỘ LỌC ĐỘNG ĐẤT (Earthquake Filter):
        - Nếu trong 2s qua giá drift > threshold → Môi trường bị phá vỡ → Yêu cầu recalculate
        - Nếu cấu trúc bị thay đổi (VD: LONG nhưng giá thủng MA) → Abort
        - Chỉ khi STATE SYNCED mới tiếp tục execution
        
        Returns:
            (is_synced, reason_if_failed)
        """
        # Extract metadata from regime calculation
        regime_meta = getattr(payload, 'regime_metadata', {})
        
        if not regime_meta:
            self.logger.warning("⚠️  No regime metadata found, skipping state sync verification")
            return (True, None)
        
        calculated_price = regime_meta.get('calculated_price')
        calculated_atr = regime_meta.get('calculated_atr')
        calculation_timestamp = regime_meta.get('calculation_timestamp')
        
        if calculated_price is None or calculated_atr is None:
            self.logger.warning("⚠️  Incomplete metadata, skipping verification")
            return (True, None)
        
        # Get current market state
        now_price = current_ticker.get('last', current_ticker.get('bid', 0))
        now_timestamp = time.time()
        
        # 1. TIME LAG CHECK
        if calculation_timestamp:
            time_lag = now_timestamp - calculation_timestamp
            
            if time_lag > time_threshold_sec:
                reason = f"STALE DATA: Calculation is {time_lag:.2f}s old (threshold: {time_threshold_sec}s). Market may have changed significantly."
                self.logger.error(f"🚨 {reason}")
                return (False, reason)
            
            self.logger.info(f"✅ Time lag check passed: {time_lag:.3f}s")
        
        # 2. PRICE DRIFT CHECK (Earthquake Filter)
        if calculated_atr > 0:
            price_drift = abs(now_price - calculated_price)
            drift_in_atr = price_drift / calculated_atr
            
            if drift_in_atr > drift_threshold_atr:
                reason = f"PRICE EARTHQUAKE: Price drifted {drift_in_atr:.2f} ATR (threshold: {drift_threshold_atr} ATR). " \
                         f"Calculated: ${calculated_price:.2f}, Now: ${now_price:.2f}, Drift: ${price_drift:.2f}"
                self.logger.error(f"🚨 {reason}")
                self.logger.error(f"   → Regime environment has been DESTROYED during calculation!")
                self.logger.error(f"   → ABORT EXECUTION. Request Regime to recalculate with new data.")
                return (False, reason)
            
            self.logger.info(f"✅ Price drift check passed: {drift_in_atr:.3f} ATR (drift: ${price_drift:.2f})")
        
        # 3. STRUCTURE VALIDITY CHECK
        # Check if market structure is still valid for the signal direction
        market_state = payload.market_state
        
        # Example: If LONG signal but price has broken below major support
        # (This requires additional market structure data from regime, simplified here)
        # In production, you'd check:
        # - LONG: price still above key EMAs/support
        # - SHORT: price still below key EMAs/resistance
        # - SIDEWAY: price still within bands
        
        # For now, we rely on price drift check as primary filter
        
        self.logger.info(f"✅ STATE SYNC VERIFIED: Market state consistent with calculation")
        self.logger.info(f"   Calculated @ T+0: ${calculated_price:.2f}")
        self.logger.info(f"   Current @ T+{time_lag:.1f}s: ${now_price:.2f}")
        self.logger.info(f"   Drift: {drift_in_atr:.3f} ATR (< {drift_threshold_atr} threshold)")
        
        return (True, None)
    
    
    def execute_payload(self, 
                       payload: ExecutionPayload, 
                       symbol: str,
                       current_ticker: Optional[Dict] = None,
                       verify_state: bool = True) -> ExecutionReport:
        
        start_time = time.time()
        
        self.logger.info(f"╔═══════════════════════════════════════════════════════════╗")
        self.logger.info(f"║  EXECUTION STARTED: {payload.strategy_name}")
        self.logger.info(f"║  Symbol: {symbol} | Market State: {payload.market_state}")
        self.logger.info(f"║  Total Lots: {payload.total_lots:.5f} | Tranches: {len(payload.tranches)}")
        self.logger.info(f"╚═══════════════════════════════════════════════════════════╝")
        
        # STATE SYNCHRONIZATION CHECK (The Ultimate Guardian)
        # Verify that market environment hasn't changed drastically during Regime calculation
        if verify_state and current_ticker:
            self.logger.info(f"\n{'═'*60}")
            self.logger.info(f"🔍 STATE SYNCHRONIZATION VERIFICATION")
            self.logger.info(f"{'═'*60}")
            
            is_synced, sync_failure_reason = self.verify_state_sync(
                payload=payload,
                current_ticker=current_ticker
            )
            
            if not is_synced:
                self.logger.error(f"\n{'🚨'*30}")
                self.logger.error(f"STATE DESYNC DETECTED!")
                self.logger.error(f"Reason: {sync_failure_reason}")
                self.logger.error(f"{'🚨'*30}\n")
                
                return ExecutionReport(
                    status=ExecutionStatus.REJECTED,
                    rejection_reason=f"STATE_DESYNC: {sync_failure_reason}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    warnings=["Market environment changed during calculation", "Recommend: Recalculate regime with fresh data"]
                )
            
            self.logger.info(f"✅ State sync verified - Environment stable\n")
        
        # MARKET CONDITIONS CHECK (Spread, Volume)
        if self.enable_live_checks and current_ticker:
            is_valid, rejection_reason = self._validate_market_conditions(
                current_ticker, payload.market_state
            )
            
            if not is_valid:
                self.logger.error(f"❌ EXECUTION REJECTED: {rejection_reason}")
                return ExecutionReport(
                    status=ExecutionStatus.REJECTED,
                    rejection_reason=rejection_reason,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        
        sequenced_tranches = self._sequence_tranches(payload.tranches)
        
        side = self._determine_order_side(payload.market_state)
        
        executed_orders = []
        failed_tranches = []
        total_filled = 0.0
        total_rejected = 0.0
        slippages = []
        
        for tranche in sequenced_tranches:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"TRANCHE {tranche.tranche_id}/{len(sequenced_tranches)}")
            self.logger.info(f"Type: {tranche.trigger_type} | Lots: {tranche.volume_lots:.5f}")
            self.logger.info(f"Entry: {tranche.entry_price} | SL: {tranche.sl_price} | TP: {tranche.tp_price}")
            self.logger.info(f"{'='*60}")
            
            try:
                order_result = self._execute_tranche(
                    tranche=tranche,
                    symbol=symbol,
                    side=side,
                    current_ticker=current_ticker
                )
                
                if order_result['status'] == 'filled' or order_result['status'] == 'open':
                    executed_orders.append(order_result)
                    total_filled += tranche.volume_lots
                    
                    if 'slippage_pct' in order_result:
                        slippages.append(order_result['slippage_pct'])
                    
                    self.logger.info(f"✅ Tranche {tranche.tranche_id} executed: Order ID {order_result.get('id', 'N/A')}")
                    
                    if tranche.trigger_type == 'MARKET':
                        fill_confirmed = self._wait_for_fill(
                            order_result.get('id'), 
                            timeout=self.config.market_order_timeout_sec
                        )
                        if not fill_confirmed:
                            self.logger.warning(f"⚠️  Market order {order_result.get('id')} not confirmed within timeout")
                else:
                    failed_tranches.append(tranche.tranche_id)
                    total_rejected += tranche.volume_lots
                    self.logger.error(f"❌ Tranche {tranche.tranche_id} failed: {order_result.get('error', 'Unknown')}")
                
            except Exception as e:
                self.logger.error(f"💥 EXCEPTION on Tranche {tranche.tranche_id}: {str(e)}")
                failed_tranches.append(tranche.tranche_id)
                total_rejected += tranche.volume_lots
                
                if self.config.enable_emergency_cancel and executed_orders:
                    self.logger.warning("🚨 Emergency protocol activated: Canceling all pending orders")
                    self._emergency_cancel_all([o['id'] for o in executed_orders if o.get('status') == 'open'])
                    break
            
            time.sleep(self.config.rate_limit_delay_ms / 1000.0)
        
        execution_time = (time.time() - start_time) * 1000
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
        
        if len(executed_orders) == len(payload.tranches):
            status = ExecutionStatus.SUCCESS
        elif len(executed_orders) > 0:
            status = ExecutionStatus.PARTIAL
        else:
            status = ExecutionStatus.FAILED
        
        report = ExecutionReport(
            status=status,
            executed_orders=executed_orders,
            failed_tranches=failed_tranches,
            total_filled_lots=total_filled,
            total_rejected_lots=total_rejected,
            execution_time_ms=execution_time,
            average_slippage_pct=avg_slippage
        )
        
        self._log_execution_summary(report, payload)
        
        return report
    
    
    def _validate_market_conditions(self, 
                                    ticker: Dict, 
                                    market_state: str) -> Tuple[bool, Optional[str]]:
        
        ask = ticker.get('ask', 0)
        bid = ticker.get('bid', 0)
        
        if ask <= 0 or bid <= 0:
            return False, "Invalid ticker data: ask/bid is zero"
        
        spread = ask - bid
        mid_price = (ask + bid) / 2
        spread_pct = spread / mid_price
        
        if spread_pct > self.config.max_spread_pct:
            return False, f"Spread too wide: {spread_pct:.4%} > {self.config.max_spread_pct:.4%} (Market Maker manipulation risk)"
        
        volume_24h = ticker.get('quoteVolume', 0)
        if volume_24h > 0 and volume_24h < 100000:
            return False, f"Low liquidity: 24h volume ${volume_24h:,.0f} (Thin order book risk)"
        
        return True, None
    
    
    def _sequence_tranches(self, tranches: List[OrderTranche]) -> List[OrderTranche]:
        
        priority_map = {
            'MARKET': 1,
            'STOP': 2,
            'LIMIT': 3
        }
        
        sorted_tranches = sorted(
            tranches, 
            key=lambda t: (priority_map.get(t.trigger_type, 99), t.tranche_id)
        )
        
        return sorted_tranches
    
    
    def _determine_order_side(self, market_state: str) -> str:
        
        if market_state == 'LONG':
            return 'buy'
        elif market_state == 'SHORT':
            return 'sell'
        else:
            return 'buy'
    
    
    def _execute_tranche(self,
                        tranche: OrderTranche,
                        symbol: str,
                        side: str,
                        current_ticker: Optional[Dict] = None) -> Dict:
        
        if self.dry_run:
            return self._simulate_order_execution(tranche, symbol, side)
        
        for attempt in range(1, self.config.order_retry_attempts + 1):
            try:
                order_params = self._format_broker_order(
                    tranche=tranche,
                    symbol=symbol,
                    side=side
                )
                
                self.logger.debug(f"Attempt {attempt}/{self.config.order_retry_attempts}: {order_params}")
                
                response = self.api.create_order(**order_params)
                
                slippage = 0.0
                if current_ticker and tranche.trigger_type == 'MARKET':
                    expected_price = tranche.entry_price
                    actual_price = response.get('average', response.get('price', expected_price))
                    slippage = abs(actual_price - expected_price) / expected_price
                
                return {
                    'id': response.get('id'),
                    'status': response.get('status'),
                    'filled': response.get('filled', 0),
                    'price': response.get('price'),
                    'average': response.get('average'),
                    'slippage_pct': slippage,
                    'tranche_id': tranche.tranche_id
                }
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed: {str(e)}")
                
                if attempt < self.config.order_retry_attempts:
                    time.sleep(self.config.retry_delay_ms / 1000.0)
                else:
                    return {
                        'status': 'failed',
                        'error': str(e),
                        'tranche_id': tranche.tranche_id
                    }
    
    
    def _format_broker_order(self, 
                            tranche: OrderTranche, 
                            symbol: str, 
                            side: str) -> Dict:
        
        order_type_map = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP': 'stop_market'
        }
        
        order_type = order_type_map.get(tranche.trigger_type, 'market')
        
        base_params = {
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': tranche.volume_lots
        }
        
        if tranche.trigger_type == 'LIMIT':
            base_params['price'] = tranche.entry_price
        
        if tranche.trigger_type == 'STOP':
            base_params['params'] = {
                'stopPrice': tranche.entry_price
            }
        
        if tranche.trigger_type == 'MARKET':
            base_params['params'] = {
                'stopLoss': {
                    'triggerPrice': tranche.sl_price
                },
                'takeProfit': {
                    'triggerPrice': tranche.tp_price
                }
            }
        
        return base_params
    
    
    def _wait_for_fill(self, order_id: str, timeout: int = 5) -> bool:
        
        if not order_id:
            return False
        
        start = time.time()
        
        while (time.time() - start) < timeout:
            try:
                order_status = self.api.fetch_order(order_id)
                
                if order_status['status'] == 'closed' or order_status['status'] == 'filled':
                    return True
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Error checking order {order_id}: {str(e)}")
                return False
        
        return False
    
    
    def _emergency_cancel_all(self, order_ids: List[str]):
        
        self.logger.warning(f"🚨 EMERGENCY CANCEL: {len(order_ids)} orders")
        
        canceled = 0
        failed = 0
        
        for oid in order_ids:
            try:
                self.api.cancel_order(oid)
                canceled += 1
                self.logger.info(f"Canceled order {oid}")
            except Exception as e:
                failed += 1
                self.logger.error(f"Failed to cancel {oid}: {str(e)}")
        
        self.logger.warning(f"Emergency cancel complete: {canceled} canceled, {failed} failed")
    
    
    def _simulate_order_execution(self, 
                                 tranche: OrderTranche, 
                                 symbol: str, 
                                 side: str) -> Dict:
        
        import random
        
        simulated_slippage = random.uniform(-0.0005, 0.0015)
        simulated_price = tranche.entry_price * (1 + simulated_slippage)
        
        self.logger.info(f"[DRY RUN] {side.upper()} {tranche.volume_lots} {symbol} @ {simulated_price}")
        
        return {
            'id': f"SIM_{tranche.tranche_id}_{int(time.time())}",
            'status': 'filled',
            'filled': tranche.volume_lots,
            'price': simulated_price,
            'average': simulated_price,
            'slippage_pct': simulated_slippage,
            'tranche_id': tranche.tranche_id
        }
    
    
    def _log_execution_summary(self, report: ExecutionReport, payload: ExecutionPayload):
        
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"EXECUTION SUMMARY: {payload.strategy_name}")
        self.logger.info(f"{'#'*60}")
        self.logger.info(f"Status: {report.status.value.upper()}")
        self.logger.info(f"Executed: {len(report.executed_orders)}/{len(payload.tranches)} tranches")
        self.logger.info(f"Filled Lots: {report.total_filled_lots:.5f}/{payload.total_lots:.5f}")
        
        if report.total_rejected_lots > 0:
            self.logger.warning(f"Rejected Lots: {report.total_rejected_lots:.5f}")
        
        if report.failed_tranches:
            self.logger.error(f"Failed Tranches: {report.failed_tranches}")
        
        self.logger.info(f"Avg Slippage: {report.average_slippage_pct:.4%}")
        self.logger.info(f"Execution Time: {report.execution_time_ms:.0f}ms")
        
        if report.rejection_reason:
            self.logger.error(f"Rejection Reason: {report.rejection_reason}")
        
        if report.warnings:
            for warning in report.warnings:
                self.logger.warning(f"⚠️  {warning}")
        
        self.logger.info(f"{'#'*60}\n")


def create_execution_commander(exchange_api, 
                               max_spread_pct: float = 0.002,
                               enable_live_checks: bool = True,
                               dry_run: bool = False) -> ExecutionCommander:
    
    config = ExecutionConfig(max_spread_pct=max_spread_pct)
    
    return ExecutionCommander(
        exchange_api=exchange_api,
        config=config,
        enable_live_checks=enable_live_checks,
        dry_run=dry_run
    )
