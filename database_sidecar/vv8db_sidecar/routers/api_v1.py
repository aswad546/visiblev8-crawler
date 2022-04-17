#import sqlalchemy as sqla
import sqlalchemy.sql as sql
import urllib.parse

from fastapi import APIRouter, HTTPException
from pydantic.dataclasses import dataclass
from urllib.parse import urlparse
from vv8db_sidecar.models.parsed_log_model import ParsedLogModel
from vv8db_sidecar.models.submission_model import SubmissionModel
from vv8db_sidecar.models.submission_response_model import SubmissionResponseModel
from vv8db_sidecar.db_conn_manager import engine
from vv8db_sidecar.util.database_util import log_entry_count



router = APIRouter(
    prefix='/api/v1'
)


# Used to create new submissions
# Returns submission id
@router.post('/submission')
async def post_submission(submission: SubmissionModel):
    print('Received url submission')
    # since this is only called from the web server we assume the url is valid
    scheme, domain, path, _, query, fragment = urlparse(submission.url)
    submission_table = sql.table(
        'submissions',
        sql.column('submission_id'),
        sql.column('url_scheme'),
        sql.column('url_domain'),
        sql.column('url_path'),
        sql.column('url_query_params'),
        sql.column('url_fragment'),
        schema='vv8_logs'
    )
    stmt = submission_table.insert().values(
        url_scheme=scheme,
        url_domain=domain,
        url_path=path,
        url_query_params=query,
        url_fragment=fragment
    ).returning(
        submission_table.c.submission_id
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(stmt)
        await conn.commit()
        ret_vals = cursor.all()
        assert len(ret_vals) == 1
        submission_id, = ret_vals[0]
    return SubmissionResponseModel(submission_id)


# Used to insert a parsed log with a given submission id
@router.post('/parsedlog')
async def post_parsed_log(parsed_log: ParsedLogModel):
    print('Received parsed log')
    submission_id = parsed_log.submission_id
    isolates_table = sql.table(
        'isolates',
        sql.column('isolate_id'),
        sql.column('isolate_value'),
        sql.column('submission_id'),
        schema='vv8_logs'
    )
    window_origin_table = sql.table(
        'window_origins',
        sql.column('window_origin_id'),
        sql.column('isolate_id'),
        sql.column('url'),
        sql.column('submission_id'),
        schema='vv8_logs'
    )
    execution_context_table = sql.table(
        'execution_contexts',
        sql.column('context_id'),
        sql.column('window_id'),
        sql.column('isolate_id'),
        sql.column('sort_index'),
        sql.column('url'),
        sql.column('script_id'),
        sql.column('src'),
        sql.column('submission_id'),
        schema='vv8_logs'
    )
    log_entry_table = sql.table(
        'log_entries',
        sql.column('sort_index'),
        sql.column('log_type'),
        sql.column('src_offset'),
        sql.column('context_id'),
        sql.column('object'),
        sql.column('function'),
        sql.column('property'),
        sql.column('arguments'),
        sql.column('submission_id'),
        schema='vv8_logs'
    )
    submission_table = sql.table(
        'submissions',
        sql.column('submission_id'),
        sql.column('end_time'),
        schema='vv8_logs'
    )

    isolate_args = [
        {
            'isolate_value': isolate.isolate_value,
            'submission_id': submission_id
        }
        for isolate in parsed_log.isolates
    ]
    

    async with engine.connect() as conn:
        # need to iterate over args since asyncpg only supports up to 32767 args in query
        arg_slice_size = 32767 // 2
        isolate_map = {}
        for i in range(0, len(isolate_args), arg_slice_size):
            args_slice = isolate_args[i:i+arg_slice_size]
            isolate_slice = parsed_log.isolates[i:i+arg_slice_size]
            stmt = (
                isolates_table.insert()
                .values(args_slice)
                .returning(isolates_table.c.isolate_id)
            )
            cursor = await conn.execute(stmt)
            all_resp = cursor.all()
            isolate_map.update({
                iso.isolate_value: iso_resp[0]
                for iso_resp, iso in zip(all_resp, isolate_slice)
            })

        window_origin_args = [
            {
                'isolate_id': isolate_map[wo.isolate_id],
                'url': wo.url,
                'submission_id': submission_id
            }
            for wo in parsed_log.window_origins
        ]
        # need to iterate over args since asyncpg only supports up to 32767 args in query
        arg_slice_size = 32767 // 3
        window_origin_map = {}
        for i in range(0, len(window_origin_args), arg_slice_size):
            args_slice = window_origin_args[i:i+arg_slice_size]
            window_origin_slice = parsed_log.window_origins[i:i+arg_slice_size]
            stmt = (
                window_origin_table.insert()
                .values(args_slice)
                .returning(window_origin_table.c.window_origin_id)
            )
            cursor = await conn.execute(stmt)
            all_resp = cursor.all()
            window_origin_map.update({
                wo.url: wo_resp[0]
                for wo_resp, wo in zip(all_resp, window_origin_slice)
            })

        execution_context_args = [
            {
                'window_id': window_origin_map[ec.window_origin],
                'isolate_id': isolate_map[ec.isolate_id],
                'sort_index': ec.sort_index,
                'url': ec.script_url,
                'script_id': ec.script_id,
                'src': ec.src,
                'submission_id': submission_id
            }
            for ec in parsed_log.execution_contexts
        ]
        # need to iterate over args since asyncpg only supports up to 32767 args
        arg_slice_size = 32767 // 7
        execution_context_map = {}
        for i in range(0, len(execution_context_args), arg_slice_size):
            args_slice = execution_context_args[i:i+arg_slice_size]
            execution_context_slice = parsed_log.execution_contexts[i:i+arg_slice_size]
            stmt = (
                execution_context_table.insert()
                .values(args_slice)
                .returning(execution_context_table.c.context_id)
            )
            cursor = await conn.execute(stmt)
            all_resp = cursor.all()
            execution_context_map.update({
                ec.script_id: ec_resp[0]
                for ec_resp, ec in zip(all_resp, execution_context_slice)
            })

        log_entry_args = [
            {
                'sort_index': le.sort_index,
                'log_type': le.log_type,
                'src_offset': le.src_offset,
                'context_id': None if le.context_id is None else execution_context_map[le.context_id],
                'object': le.obj,
                'function': le.func,
                'property': le.prop,
                'arguments': None if le.args is None else ':'.join(le.args),
                'submission_id': submission_id
            }
            for le in parsed_log.log_entries
        ]
        # need to iterate over args since asyncpg only supports up to 32767 args
        arg_slice_size = 32767 // 9
        for i in range(0, len(log_entry_args), arg_slice_size):
            args_slice = log_entry_args[i:i+arg_slice_size]
            log_entry_slice = parsed_log.log_entries[i:i+arg_slice_size]
            stmt = log_entry_table.insert().values(args_slice)
            await conn.execute(stmt)

        update_sub_stmt = (
            submission_table.update()
            .values(end_time=sql.functions.current_timestamp())
            .where(submission_table.c.submission_id==submission_id)
        )
        await conn.execute(update_sub_stmt)

        await conn.commit()


@dataclass
class SubmissionIdExistsResponse:
    submission_id: int
    exists: bool


# used to check if a given submission id exists
@router.get('/submission/{submission_id}/exists', response_model=SubmissionIdExistsResponse)
async def get_submission_ids(submission_id: int):
    submission_table = sql.table(
        'submissions',
        sql.column('submission_id'),
        schema='vv8_logs'
    )
    select_stmt = (
        submission_table.select()
        .where(submission_table.c.submission_id==submission_id)
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(select_stmt)
        all_resp = cursor.all()
        if len(all_resp) == 0:
            # No submission found
            raise HTTPException(status_code=404, detail="Submission not found")
        else:
            # found submission
            assert len(all_resp) == 1
            assert all_resp[0][0] == submission_id
            return SubmissionIdExistsResponse(submission_id, True)


@dataclass
class RecentSubmissionResponse:
    submission_id: int | None


# Used to get the most recent submission id for a given url
@router.get('/submission', response_model=RecentSubmissionResponse)
async def get_recent_submission(url: str):
    print('GET SUBMISSION', url)
    raw_url = urllib.parse.unquote(url)
    scheme, domain, path, _, query, fragment = urllib.parse.urlparse(raw_url)
    query_params = {
        'url_scheme': scheme,
        'url_domain': domain,
        'url_path': path,
        'url_query_params': query,
        'url_fragment': fragment
    }
    select_stmt = sql.text('''
        SELECT submission_id
        FROM vv8_logs.submissions s
        WHERE
            s.url_scheme = :url_scheme
            AND s.url_domain = :url_domain
            AND s.url_path = :url_path
            AND s.url_query_params = :url_query_params
            AND s.url_fragment = :url_fragment
        ORDER BY s.start_time DESC
        LIMIT 1;
    ''')
    async with engine.connect() as conn:
        cursor = await conn.execute(select_stmt, query_params)
        all_resp = cursor.all()
    if len(all_resp) == 0:
        return RecentSubmissionResponse(None)
    elif len(all_resp) == 1:
        return RecentSubmissionResponse(all_resp[0][0])
    else:
        raise HTTPException(status_code=500)


@router.get('/submission/{submission_id}/gets')
async def get_submission_id_gets(submission_id: int):
    log_entry_table = sql.table(
        'log_entries',
        sql.column('log_entry_id'),
        sql.column('submission_id'),
        sql.column('sort_index'),
        sql.column('log_type'),
        sql.column('src_offset'),
        sql.column('context_id'),
        sql.column('object'),
        sql.column('property'),
        schema='vv8_logs'
    )
    stmt = (
        log_entry_table.select()
        .where(
            log_entry_table.c.submission_id==submission_id,
            log_entry_table.c.log_type=='get')
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(stmt)
        all_resp = cursor.mappings().all()
    return all_resp


@router.get('/submission/{submission_id}/gets/count')
async def get_submission_id_gets_count(submission_id: int):
    return await log_entry_count(submission_id, 'get')


@router.get('/submission/{submission_id}/sets')
async def get_submission_id_sets(submission_id: int):
    log_entry_table = sql.table(
        'log_entries',
        sql.column('log_entry_id'),
        sql.column('submission_id'),
        sql.column('sort_index'),
        sql.column('log_type'),
        sql.column('src_offset'),
        sql.column('context_id'),
        sql.column('object'),
        sql.column('property'),
        sql.column('arguments'),
        schema='vv8_logs'
    )
    stmt = (
        log_entry_table.select()
        .where(
            log_entry_table.c.submission_id==submission_id,
            log_entry_table.c.log_type=='set')
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(stmt)
        all_resp = cursor.mappings().all()
    return all_resp


@router.get('/submission/{submission_id}/sets/count')
async def get_submission_id_sets_count(submission_id: int):
    return await log_entry_count(submission_id, 'set')


@router.get('/submission/{submission_id}/constructions')
async def get_submission_id_constructions(submission_id: int):
    log_entry_table = sql.table(
        'log_entries',
        sql.column('log_entry_id'),
        sql.column('submission_id'),
        sql.column('sort_index'),
        sql.column('log_type'),
        sql.column('src_offset'),
        sql.column('context_id'),
        sql.column('function'),
        sql.column('arguments'),
        schema='vv8_logs'
    )
    stmt = (
        log_entry_table.select()
        .where(
            log_entry_table.c.submission_id==submission_id,
            log_entry_table.c.log_type=='new')
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(stmt)
        all_resp = cursor.mappings().all()
    output = [
        dict(row)
        for row in all_resp
    ]
    for x in output:
        x['arguments'] = x['arguments'].split(':')
    return all_resp


@router.get('/submission/{submission_id}/constructions/count')
async def get_submission_id_constructions_count(submission_id: int):
    return await log_entry_count(submission_id, 'new')


@router.get('/submission/{submission_id}/calls')
async def get_submission_id_calls(submission_id: int):
    log_entry_table = sql.table(
        'log_entries',
        sql.column('log_entry_id'),
        sql.column('submission_id'),
        sql.column('sort_index'),
        sql.column('log_type'),
        sql.column('src_offset'),
        sql.column('context_id'),
        sql.column('object'),
        sql.column('function'),
        sql.column('arguments'),
        schema='vv8_logs'
    )
    stmt = (
        log_entry_table.select()
        .where(
            log_entry_table.c.submission_id==submission_id,
            log_entry_table.c.log_type=='call')
    )
    async with engine.connect() as conn:
        cursor = await conn.execute(stmt)
        all_resp = cursor.mappings().all()
    output = [
        dict(row)
        for row in all_resp
    ]
    for x in output:
        x['arguments'] = x['arguments'].split(':')
    return output


@router.get('/submission/{submission_id}/calls/count')
async def get_submission_id_calls_count(submission_id: int):
    return await log_entry_count(submission_id, 'call')
