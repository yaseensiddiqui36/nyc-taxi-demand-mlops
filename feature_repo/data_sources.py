from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

hourly_rides_source = PostgreSQLSource(
    name="hourly_rides_pg",
    query="SELECT pickup_location_id, pickup_hour AS event_timestamp, ride_count FROM public.hourly_rides",
    timestamp_field="event_timestamp",
    description="Raw hourly taxi ride counts from NYC TLC data",
)
