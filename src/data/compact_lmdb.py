#!/usr/bin/env python3
"""
LMDB Database Compaction Script
Run this after preprocessing to shrink oversized sparse files

Usage:
    python compact_lmdb.py /path/to/database --overwrite
    python compact_lmdb.py /path/to/database --verify  # Slower but checks integrity
"""

import lmdb
import os
import shutil
import argparse
import hashlib
from pathlib import Path


def get_actual_size(path):
    """Get the real disk usage (not logical size) on Linux/Windows"""
    try:
        # st_blocks is 512-byte blocks on Unix
        return os.stat(path).st_blocks * 512
    except AttributeError:
        # Windows fallback - just return file size
        return os.path.getsize(path)


def verify_databases(src_env, dst_env):
    """
    Verify that all keys and values match between source and destination.
    This is slow but ensures data integrity.
    """
    print("\nVerifying database integrity...")

    with src_env.begin() as src_txn:
        with dst_env.begin() as dst_txn:
            src_cursor = src_txn.cursor()
            dst_cursor = dst_txn.cursor()

            # Check entry counts match
            src_count = sum(1 for _ in src_cursor)
            dst_count = sum(1 for _ in dst_cursor)

            if src_count != dst_count:
                raise ValueError(f"Entry count mismatch: {src_count} vs {dst_count}")

            print(f"  Entry count verified: {src_count}")

            # Check all key-value pairs match
            src_cursor.first()
            dst_cursor.first()

            checked = 0
            mismatches = []

            while True:
                src_item = src_cursor.item()
                dst_item = dst_cursor.item()

                if src_item != dst_item:
                    mismatches.append((src_item[0][:50], len(src_item[1]), len(dst_item[1])))
                    if len(mismatches) >= 5:
                        break

                checked += 1
                if checked % 10000 == 0:
                    print(f"  Verified {checked}/{src_count} entries...", end='\r')

                src_next = src_cursor.next()
                dst_next = dst_cursor.next()

                if src_next != dst_next:
                    raise ValueError("Cursor mismatch - databases have different structure")

                if not src_next:
                    break

            if mismatches:
                print(f"\n  ERROR: Found {len(mismatches)} mismatches!")
                for key_preview, src_len, dst_len in mismatches[:5]:
                    print(f"    Key: {key_preview}... (src={src_len}, dst={dst_len})")
                return False

            print(f"\n  All {checked} entries verified successfully!")
            return True


def compact_lmdb(src_dir, dst_dir=None, overwrite=False, verify=False):
    """
    Compact an LMDB database by copying only actual data to a new database
    with right-sized map allocation.

    Args:
        src_dir: Source LMDB directory (contains data.mdb and lock.mdb)
        dst_dir: Destination directory (default: src_dir + '_compact')
        overwrite: Whether to overwrite source after compaction
        verify: Whether to verify all data copied correctly (slower)
    """
    src_path = Path(src_dir)
    src_data = src_path / 'data.mdb'
    src_lock = src_path / 'lock.mdb'

    if not src_data.exists():
        raise FileNotFoundError(f"No data.mdb found in {src_dir}")

    # Default destination
    if dst_dir is None:
        dst_dir = str(src_path) + '_compact'

    dst_path = Path(dst_dir)

    # Don't overwrite existing destination unless --overwrite
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination {dst_dir} exists. Use --overwrite to replace.")

    dst_path.mkdir(parents=True, exist_ok=True)

    dst_data = dst_path / 'data.mdb'
    dst_lock = dst_path / 'lock.mdb'

    print(f"Source: {src_dir}")
    logical_size = src_data.stat().st_size
    actual_size = get_actual_size(src_data)
    print(f"  Logical size: {logical_size / 1e9:.2f} GB")
    print(f"  Actual size:  {actual_size / 1e9:.2f} GB")
    print(f"  Sparse ratio: {(1 - actual_size/logical_size)*100:.1f}%")
    print()

    # Open source read-only with no lock (we're not writing to it)
    src_env = lmdb.open(str(src_path), readonly=True, lock=False)

    # Get stats for sizing
    stats = src_env.stat()
    entries = stats['entries']
    psize = stats['psize']

    print(f"Database stats:")
    print(f"  Entries: {entries:,}")
    print(f"  Page size: {psize}")
    print(f"  Tree depth: {stats['depth']}")
    print()

    # Calculate new map size
    # Method: actual data size + B-tree overhead + 20% margin
    # B-tree typically adds 50-100% overhead for indexing

    # Sample some entries to estimate average size
    sample_size = min(1000, entries)
    total_sample_size = 0

    with src_env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= sample_size:
                break
            total_sample_size += len(key) + len(value)

    avg_entry_size = total_sample_size / sample_size if sample_size > 0 else 1024
    estimated_raw_data = avg_entry_size * entries

    # Conservative estimate: raw data * 2.5 for B-tree + 20% margin
    estimated_map = int(estimated_raw_data * 3.0)
    # Round up to nearest 100MB for cleanliness
    estimated_map = ((estimated_map // 104_857_600) + 1) * 104_857_600

    # Ensure minimum 10MB
    estimated_map = max(estimated_map, 10 * 1024 * 1024)

    print(f"Size estimation:")
    print(f"  Sampled {sample_size} entries, avg {avg_entry_size:.0f} bytes each")
    print(f"  Estimated raw data: {estimated_raw_data / 1e9:.2f} GB")
    print(f"  New map size: {estimated_map / 1e9:.2f} GB (with overhead)")
    print()

    # Create compacted database
    dst_env = lmdb.open(str(dst_path), map_size=estimated_map)

    print("Copying data...")
    with src_env.begin() as src_txn:
        cursor = src_txn.cursor()
        with dst_env.begin(write=True) as dst_txn:
            batch_size = 1000
            batch = []
            copied = 0

            for key, value in cursor:
                batch.append((key, value))

                if len(batch) >= batch_size:
                    for k, v in batch:
                        dst_txn.put(k, v)
                    batch = []
                    copied += batch_size
                    if copied % 10000 == 0:
                        print(f"  Copied {copied:,}/{entries:,} entries...", end='\r')

            # Final batch
            for k, v in batch:
                dst_txn.put(k, v)

            dst_txn.commit()

    print(f"\n  Copied {entries:,} entries total")

    # Verify if requested
    if verify:
        if not verify_databases(src_env, dst_env):
            print("\nVerification FAILED! Keeping original, removing compacted version.")
            src_env.close()
            dst_env.close()
            shutil.rmtree(dst_path)
            return None

    src_env.close()
    dst_env.close()

    # Copy lock file (usually tiny, may not exist on some platforms)
    if src_lock.exists():
        shutil.copy2(src_lock, dst_lock)

    # Final stats
    new_logical = dst_data.stat().st_size
    new_actual = get_actual_size(dst_data)

    print(f"\nCompaction complete!")
    print(f"Destination: {dst_dir}")
    print(f"  New logical size: {new_logical / 1e9:.2f} GB")
    print(f"  New actual size:  {new_actual / 1e9:.2f} GB")
    print(f"  Size reduction:   {(1 - new_logical/logical_size)*100:.1f}%")
    print()

    if overwrite:
        print(f"Overwriting original...")
        backup_dir = str(src_path) + '_backup_' + str(int(os.time()))

        # Rename original to backup (faster than copy)
        os.rename(str(src_path), backup_dir)
        os.rename(str(dst_path), str(src_path))

        print(f"  Original backed up to: {backup_dir}")
        print(f"  Compacted database now at: {src_dir}")

        # Verify final
        final_size = (src_path / 'data.mdb').stat().st_size
        print(f"  Final size: {final_size / 1e9:.2f} GB")

        return src_dir

    return dst_dir


def main():
    parser = argparse.ArgumentParser(
        description='Compact LMDB sparse files after bulk loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create compacted copy in new directory
  python compact_lmdb.py /data/mydb

  # Replace original with compacted version (keeps backup)
  python compact_lmdb.py /data/mydb --overwrite

  # Verify all data after copying (slower but safer)
  python compact_lmdb.py /data/mydb --verify

  # Specify custom output location
  python compact_lmdb.py /data/mydb -o /data/mydb_small
        """
    )
    parser.add_argument(
        'source',
        help='Source LMDB directory (containing data.mdb)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: source_compact)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace original with compacted version (creates timestamped backup)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify all keys and values match after copying (slower)'
    )

    args = parser.parse_args()

    try:
        result = compact_lmdb(args.source, args.output, args.overwrite, args.verify)
        if result:
            print(f"\nSuccess! Compacted database at: {result}")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    main()