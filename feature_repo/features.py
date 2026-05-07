from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Int32
from feast.value_type import ValueType

from data_sources import hourly_rides_source

taxi_zone = Entity(
    name="pickup_location_id",
    description="NYC Taxi Zone ID (1–263)",
    join_keys=["pickup_location_id"],
    value_type=ValueType.INT64,
)

hourly_rides_view = FeatureView(
    name="hourly_rides_view",
    entities=[taxi_zone],
    ttl=timedelta(days=30),
    schema=[
        Field(name="ride_count", dtype=Int32),
    ],
    source=hourly_rides_source,
    description="Hourly taxi ride counts per NYC zone",
    tags={"team": "mlops", "project": "taxi_demand"},
)
