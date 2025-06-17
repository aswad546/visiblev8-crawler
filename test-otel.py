from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Install requirements if needed:
# pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

# Configure the exporter to send to your collector
resource = Resource(attributes={"service.name": "test-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create a tracer and a test span
tracer = trace.get_tracer("test-tracer")
with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test.attribute", "test-value")
    print("Sent test span to OTEL collector")

# Make sure to flush all pending spans
trace.get_tracer_provider().force_flush()