def class_per_data_spliter(binds, chunk_size, parquet_path, ctd_df, process_row, output_path, max_chunk=0):
    offset = 0
    chunk_number = 0
    num_workers = cpu_count()
    pool = Pool(num_workers)

    con = duckdb.connect()
    while True:
        if max_chunk and chunk_number == max_chunk-1:
            break

        start_time = time.time()
        
        chunk = con.execute(f"""
        SELECT *
        FROM parquet_scan('{parquet_path}')
        WHERE binds = {binds}
        LIMIT {chunk_size} 
        OFFSET {offset}
        """).fetch_df()

        if chunk.empty:
            break

        smiles_list = chunk['molecule_smiles'].tolist()
        
        ## 병렬로 데이터 처리
        results = pool.map(process_row, smiles_list)

        ## 결과를 데이터프레임으로 변환
        fingerprints = [result['fingerprint'] for result in results]
        descriptors_list = [result['descriptors'] for result in results]
        
        chunk['fingerprints'] = fingerprints
        descriptor_df = pd.DataFrame(descriptors_list)
        excluded_descriptors = descriptor_df.columns[descriptor_df.isna().any()].tolist()
        descriptor_df.drop(columns=excluded_descriptors, inplace=True)
        used_descriptor = descriptor_df.columns.tolist()

        if chunk_number == 0:
            print(f"제외된 descriptors: {excluded_descriptors}")
            print(f"사용된 descriptors: {used_descriptor}")

        ## CTD 데이터 병합 (protein_name 기준)
        merged_chunk = pd.merge(chunk, ctd_df, on='protein_name', how='left')
        merged_chunk = pd.concat([merged_chunk, descriptor_df], axis=1)
        
        table = pa.Table.from_pandas(merged_chunk)

        output_file = f"{output_path}/b{binds}_chunk_{chunk_number}.parquet"
        pq.write_table(table, output_file)

        offset += chunk_size
        chunk_number += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processed chunk: {chunk_number}, offset: {offset} saved to {output_file}. Time taken: {elapsed_time:.2f} seconds")

    pool.close()
    pool.join()
    con.close()
