import influxdb_client
import pandas as pd
import os

from data_and_connection import write_data


def delete_data(url, token, organization, bucket):
    """
    Deletes all data from a specified bucket.
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: None
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.delete_api()
    api.delete(bucket=f'"{bucket}"', org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="*"')

    print("Deleted ok")


def read_data(url, token, organization, bucket):
    """
    Reads data from a specified bucket and stores it in a DataFrame
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: The DataFrame
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.query_api()
    query = f'from(bucket: "{bucket}") |> range(start: 0)'
    data = api.query(org=organization, query=query)
    print("Read ok")

    # Extract the records from the result
    records = []
    for table in data:
        for record in table.records:
            print(f'{record["_field"]} || {record["_value"]}')
            records.append(record.values)

    # Create a list to store the unique "field" values
    unique_fields = list(set(record["_field"] for record in records))
    print(len(unique_fields))
    d = dict()

    # Add a column for each unique "field" value
    for name in unique_fields:
        name_data = [record["_value"] for record in records if record["_field"] == name]
        d[name] = name_data

    df = pd.DataFrame(d)
    df.to_csv("./test/sample_output.csv", index=False)

    return df


if __name__ == "__main__":
    # Load 'sample input'
    sample_input = pd.read_csv("./test/sample_input.csv")

    # Import credentials
    url = os.environ.get('Url_influx_db')
    org = os.environ.get('Organization_influx')
    auth_token = os.environ.get('Auth_token')
    bucket = os.environ.get('Bucket')

    # Wipe clean the test bucket
    delete_data(url, auth_token, org, bucket)
    
    # Read each data sample and write it to the test bucket
    # Function 'write_data' is imported from the ingestion script and does the preprocessing etc
    for _, row in sample_input.iterrows():
        write_data(row, url, auth_token, org, bucket)

    # Get the 'sample output' by querying the test bucket
    sample_output = read_data(url, auth_token, org, bucket)
