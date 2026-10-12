"""Validate inventory and pseudonymize locally. Never log sensitive field values."""
import hashlib
import hmac
import re
import unicodedata
from zoneinfo import ZoneInfo
import pandas as pd

REQUIRED_COLUMNS = ['message_id','display_name','timestamp','message','message_type','topic','is_question','speaker_role']

class InventoryValidationError(ValueError):
    pass

def normalized_name(name):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(name))).strip().casefold()

def participant_id(name, key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise InventoryValidationError('Pseudonym key must contain at least 32 bytes.')
    return 'P-' + hmac.new(key, normalized_name(name).encode('utf-8'), hashlib.sha256).hexdigest()

def validate_inventory(frame, *, assume_timezone=None, deduplicate_exact=False):
    if assume_timezone is not None:
        try:
            if not isinstance(assume_timezone,str) or (assume_timezone!='UTC' and '/' not in assume_timezone) or assume_timezone not in __import__('zoneinfo').available_timezones():
                raise ValueError('Not an IANA zone')
            ZoneInfo(assume_timezone)
        except Exception:
            raise InventoryValidationError('An explicitly confirmed valid IANA source timezone is required.') from None
    if set(frame.columns) != set(REQUIRED_COLUMNS) or frame.columns.duplicated().any():
        raise InventoryValidationError('Inventory must contain exactly the eight documented columns; no duplicated or unknown columns.')
    result = frame.copy().fillna('').astype(str)
    for column in REQUIRED_COLUMNS:
        result[column] = result[column].str.strip()
    for column in ['message_id','display_name','timestamp','message']:
        missing = result[column].eq('')
        if missing.any():
            raise InventoryValidationError(f'Required {column} values are missing in {int(missing.sum())} records.')
    # Record numbers refer to parsed CSV records, not physical lines in multiline CSVs.
    result['source_record_numbers'] = [str(i+1) for i in range(len(result))]
    duplicate_count = 0
    if result.message_id.duplicated().any():
        if not deduplicate_exact:
            raise InventoryValidationError('Duplicate message IDs found. Exact duplicate removal must be explicitly requested.')
        unique_rows = []
        for _, group in result.groupby('message_id', sort=False):
            if len(group[REQUIRED_COLUMNS].drop_duplicates()) != 1:
                raise InventoryValidationError('Conflicting records share a message ID; no automatic deduplication is safe.')
            row = group.iloc[0].copy()
            row['source_record_numbers'] = ','.join(group.source_record_numbers)
            unique_rows.append(row)
            duplicate_count += len(group)-1
        result = pd.DataFrame(unique_rows).reset_index(drop=True)
    parsed = []
    for timestamp in result.timestamp:
        # Require an unambiguous ISO date rather than locale-dependent day/month guessing.
        if not re.match(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', timestamp):
            raise InventoryValidationError('Timestamps must be valid ISO-8601 datetimes.')
        try:
            value = pd.Timestamp(timestamp)
            if pd.isna(value):
                raise ValueError('NaT')
            if value.tzinfo is None:
                if not assume_timezone:
                    raise InventoryValidationError('Naive timestamps require an explicit --assume-timezone.')
                value = value.tz_localize(assume_timezone, ambiguous='raise', nonexistent='raise')
            parsed.append(value.tz_convert('UTC').isoformat())
        except InventoryValidationError:
            raise
        except Exception:
            raise InventoryValidationError('Invalid timestamp, timezone, or ambiguous/nonexistent local time.') from None
    result['timestamp_utc'] = pd.Series(parsed, index=result.index, dtype='object')
    labels = result.is_question.str.casefold()
    allowed = {'true','false','1','0','yes','no','', 'unknown', 'unclear'}
    if not labels.isin(allowed).all():
        raise InventoryValidationError('Preliminary is_question must be boolean-like or blank/unknown; it is not ground truth.')
    result['preliminary_question'] = labels.map({'true':True,'1':True,'yes':True,'false':False,'0':False,'no':False}).astype('object')
    result['metadata_review'] = result[['message_type','topic','speaker_role']].eq('').any(axis=1) | result.preliminary_question.isna()
    return result, {'input_records':len(frame), 'unique_messages':len(result), 'exact_duplicates_removed':duplicate_count}

def prepare_messages(inventory, key, mode):
    if mode not in {'synthetic','real'}:
        raise InventoryValidationError('Mode must be synthetic or real.')
    result = pd.DataFrame({'message_id':inventory.message_id,
                          'participant_id':[participant_id(name,key) for name in inventory.display_name],
                          'text':inventory.message, 'source_topic':''})
    result.attrs['synthetic'] = mode == 'synthetic'
    result.attrs['data_mode'] = mode
    return result
