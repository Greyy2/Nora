"""
Post-Processing Module
Analyze results after optimization completes
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
from database.mongo_service import MongoService
from optimize.sensitivity import load_valid_strategies, get_top_strategies, get_elite_strategies


def analyze_results(batch_id: str, mongo: MongoService, progress_callback: Optional[Callable] = None, collection_type: str = 'backtest') -> Dict[str, Any]:
    """
    Analyze optimization results
    
    Steps:
    1. Load all valid strategies
    2. Calculate overall statistics
    3. Find top strategies
    4. Prepare chart data
    
    Args:
        batch_id: Batch ID
        mongo: MongoService instance
        progress_callback: Optional callback to report progress (message)
        collection_type: 'backtest' or 'wfo'
        
    Returns:
        Analysis results dict
    """
    def _notify(message: str):
        """Helper to notify progress"""
        if progress_callback:
            try:
                progress_callback({'message': message})
            except Exception as e:
                print(f"⚠️  Progress callback error: {e}")
    
    print(f"\n📊 STEP 7: POST-PROCESSING - Phân tích kết quả")
    print(f"="*70)
    _notify("🔍 Đang tải dữ liệu chiến thuật...")
    
    # Load valid strategies
    strategies = load_valid_strategies(batch_id, mongo, collection_type)
    
    if not strategies:
        print("⚠️  Không có chiến thuật hợp lệ để phân tích")
        _notify("⚠️ Không có chiến thuật hợp lệ")
        return {
            'total_strategies': 0,
            'valid_strategies': 0,
            'stats': {},
            'top_strategies': []
        }
    
    print(f"✓ Đã tải {len(strategies):,} chiến thuật hợp lệ")
    _notify(f"✓ Đã tải {len(strategies):,} chiến thuật - Đang tính toán thống kê...")
    
    # Extract metrics
    rois = []
    mdds = []
    sharpes = []
    total_returns = []
    win_rates = []
    
    for s in strategies:
        result = s.get('result', {})
        if 'roi' in result:
            rois.append(result['roi'])
        if 'max_drawdown_pct' in result:
            mdds.append(abs(result['max_drawdown_pct']))
        if 'sharpe' in result:
            sharpes.append(result['sharpe'])
        if 'total_return' in result:
            total_returns.append(result['total_return'])
        if 'win_rate' in result:
            win_rates.append(result['win_rate'])
    
    # Calculate statistics
    stats = {}
    
    if rois:
        stats['roi'] = {
            'mean': float(np.mean(rois)),
            'median': float(np.median(rois)),
            'std': float(np.std(rois)),
            'min': float(np.min(rois)),
            'max': float(np.max(rois)),
            'positive_count': sum(1 for r in rois if r > 0),
            'negative_count': sum(1 for r in rois if r < 0)
        }
    
    if mdds:
        stats['mdd'] = {
            'mean': float(np.mean(mdds)),
            'median': float(np.median(mdds)),
            'std': float(np.std(mdds)),
            'min': float(np.min(mdds)),
            'max': float(np.max(mdds))
        }
    
    if sharpes:
        stats['sharpe'] = {
            'mean': float(np.mean(sharpes)),
            'median': float(np.median(sharpes)),
            'std': float(np.std(sharpes)),
            'min': float(np.min(sharpes)),
            'max': float(np.max(sharpes)),
            'positive_count': sum(1 for s in sharpes if s > 0)
        }
    
    if win_rates:
        stats['win_rate'] = {
            'mean': float(np.mean(win_rates)),
            'median': float(np.median(win_rates)),
            'std': float(np.std(win_rates)),
            'min': float(np.min(win_rates)),
            'max': float(np.max(win_rates))
        }
    
    print(f"\n📈 Thống kê tổng thể:")
    if 'roi' in stats:
        print(f"   ROI: {stats['roi']['mean']:.2f}% (±{stats['roi']['std']:.2f}%)")
        print(f"        Min: {stats['roi']['min']:.2f}% | Max: {stats['roi']['max']:.2f}%")
        print(f"        Positive: {stats['roi']['positive_count']:,}/{len(rois):,} ({stats['roi']['positive_count']/len(rois)*100:.1f}%)")
    
    if 'sharpe' in stats:
        print(f"   Sharpe: {stats['sharpe']['mean']:.3f} (±{stats['sharpe']['std']:.3f})")
        print(f"        Min: {stats['sharpe']['min']:.3f} | Max: {stats['sharpe']['max']:.3f}")
    
    if 'mdd' in stats:
        print(f"   MDD: {stats['mdd']['mean']:.2f}% (±{stats['mdd']['std']:.2f}%)")
    
    _notify("📈 Hoàn tất thống kê - Đang tìm top strategies...")
    
    # Find top strategies
    print(f"\n🏆 Tìm chiến thuật tốt nhất...")
    
    top_by_sharpe = get_top_strategies(batch_id, mongo, top_n=10, sort_by='sharpe', collection_type=collection_type)
    top_by_roi = get_top_strategies(batch_id, mongo, top_n=10, sort_by='roi', collection_type=collection_type)
    
    print(f"   Top 10 by Sharpe: {len(top_by_sharpe)} strategies")
    if top_by_sharpe:
        best_sharpe = top_by_sharpe[0]['result']
        print(f"   #1 Sharpe: {best_sharpe.get('sharpe', 0):.3f} (ROI: {best_sharpe.get('roi', 0):.2f}%)")
    
    print(f"   Top 10 by ROI: {len(top_by_roi)} strategies")
    if top_by_roi:
        best_roi = top_by_roi[0]['result']
        print(f"   #1 ROI: {best_roi.get('roi', 0):.2f}% (Sharpe: {best_roi.get('sharpe', 0):.3f})")
    
    # 🏆 Elite Strategies (Strict Selection)
    print(f"\n💎 Tìm Top 20 Elite (Khắt khe)...")
    elite_hashes = get_elite_strategies(batch_id, mongo, max_top=20, collection_type=collection_type)
    print(f"   Tìm được {len(elite_hashes)} chiến thuật đạt chuẩn Elite")
    
    if elite_hashes and strategies:
        # For logging purposes, find the first elite strategy in the loaded list
        first_elite = next((s for s in strategies if s.get('config_hash') == elite_hashes[0]), None)
        if first_elite:
            res = first_elite.get('result', {})
            print(f"   #1 Elite: ROI: {res.get('roi', 0):.2f}% | MDD: {abs(res.get('max_drawdown_pct', 0)):.2f}% | Lev: {res.get('max_leverage', 0):.1f}x")
    
    _notify("🏆 Hoàn tất tìm top strategies - Đang chuẩn bị biểu đồ...")
    
    # Prepare chart data
    print(f"\n📊 Chuẩn bị dữ liệu biểu đồ...")
    chart_data = []
    for s in strategies:
        result = s.get('result', {})
        if 'roi' in result and 'max_drawdown_pct' in result:
            chart_data.append({
                'id': str(s['_id']),
                'roi': result['roi'],
                'mdd': abs(result['max_drawdown_pct']),
                'sharpe': result.get('sharpe', 0),
                'total_return': result.get('total_return', 0)
            })
    
    print(f"   Đã chuẩn bị {len(chart_data):,} điểm dữ liệu cho biểu đồ")
    
    _notify(f"✅ Hoàn tất phân tích {len(strategies):,} chiến thuật!")
    
    print(f"="*70)
    print(f"✅ POST-PROCESSING hoàn tất!")
    print(f"="*70)
    
    return {
        'total_strategies': len(strategies),
        'valid_strategies': len(strategies),
        'stats': stats,
        'top_by_sharpe': [str(s['_id']) for s in top_by_sharpe[:5]],
        'top_by_roi': [str(s['_id']) for s in top_by_roi[:5]],
        'elite_strategies': elite_hashes,  # 🏹 List of config_hashes
        'chart_data_count': len(chart_data)
    }
