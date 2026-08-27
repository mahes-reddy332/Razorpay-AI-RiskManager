import pytest
import networkx as nx
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tracer import get_chain_metrics

def test_get_chain_metrics_linear_mule():
    """Tests if the tracer correctly identifies a rapid linear mule chain."""
    G = nx.MultiDiGraph()
    base_time = datetime(2026, 6, 1, 10, 0)
    
    G.add_edge('VICTIM', 'MULE1', amount=1000, timestamp=base_time, txn_id='1')
    G.add_edge('MULE1', 'MULE2', amount=1000, timestamp=base_time + timedelta(minutes=15), txn_id='2')
    G.add_edge('MULE2', 'CASHOUT', amount=1000, timestamp=base_time + timedelta(minutes=30), txn_id='3')
    
    # Trace from intermediate node MULE1
    hops, nodes, _ = get_chain_metrics(G, 'MULE1', max_hours=24)
    assert hops == 3  # VICTIM->MULE1, MULE1->MULE2, MULE2->CASHOUT
    assert nodes == 4 # VICTIM, MULE1, MULE2, CASHOUT

def test_get_chain_metrics_temporal_break():
    """Tests if the tracer correctly ignores transactions outside the rapid flow window (e.g. Salary spend)."""
    G = nx.MultiDiGraph()
    base_time = datetime(2026, 6, 1, 10, 0)
    
    # Corp pays Salary
    G.add_edge('CORP', 'EMP', amount=50000, timestamp=base_time, txn_id='1')
    # Emp buys groceries 48 hours later (breaks 24h tracer limit)
    G.add_edge('EMP', 'MERCHANT', amount=1000, timestamp=base_time + timedelta(hours=48), txn_id='2')
    
    # Trace from EMP
    hops, nodes, _ = get_chain_metrics(G, 'EMP', max_hours=24)
    
    # Should only trace the inbound, ignoring the late outbound
    assert hops == 1
    assert nodes == 2 # CORP, EMP
    
def test_get_chain_metrics_fan_out():
    """Tests if the tracer correctly counts a wide fan-out (e.g. SMB or Structuring)."""
    G = nx.MultiDiGraph()
    base_time = datetime(2026, 6, 1, 10, 0)
    
    G.add_edge('EXT', 'MULE_HUB', amount=10000, timestamp=base_time, txn_id='1')
    
    for i in range(5):
        G.add_edge('MULE_HUB', f'CASHOUT_{i}', amount=2000, 
                   timestamp=base_time + timedelta(minutes=10 + i), txn_id=f'OUT_{i}')
                   
    hops, nodes, _ = get_chain_metrics(G, 'MULE_HUB', max_hours=24)
    assert hops == 2 # EXT->HUB, HUB->CASHOUTs (All outbounds are 1 hop depth from HUB)
    assert nodes == 7 # EXT, HUB, and 5 CASHOUTs
