from common import get_partition

def test_get_partition_is_consistent():
    partition1 = get_partition()
    partition2 = get_partition()
    assert partition1 == partition2
    
if __name__ == "__main__":
    test_get_partition_is_consistent()
