// Harness-only addition. Production TraceZip source remains unchanged.
package prefix_compressed_receiver
import (
 "bytes"
 "compress/gzip"
 "context"
 "encoding/binary"
 "encoding/json"
 "fmt"
 "net/http/httptest"
 "os"
 "path/filepath"
 "reflect"
 "runtime"
 "strconv"
 "strings"
 "testing"
 "time"
 "angrychow/otel/prefix-compressed-receiver/internal/trace"
 "go.opentelemetry.io/collector/component"
 "go.opentelemetry.io/collector/consumer"
 "go.opentelemetry.io/collector/pdata/pcommon"
 "go.opentelemetry.io/collector/pdata/ptrace"
 "go.opentelemetry.io/collector/pdata/ptrace/ptraceotlp"
 "go.opentelemetry.io/collector/receiver"
 "go.opentelemetry.io/collector/receiver/receiverhelper"
 metricnoop "go.opentelemetry.io/otel/metric/noop"
 tracenoop "go.opentelemetry.io/otel/trace/noop"
 "go.uber.org/zap"
)
func TestHQSBAdmission(t *testing.T) {
 mode:=os.Getenv("HQSB_MODE");shape:=os.Getenv("HQSB_SHAPE");profile,_:=strconv.Atoi(os.Getenv("HQSB_PROFILE"))
 n:=512; td:=ptrace.NewTraces();r:=td.ResourceSpans().AppendEmpty();r.Resource().Attributes().PutStr("host.name","hqsb");r.Resource().Attributes().PutStr("service.name","hqsb")
 scope:=r.ScopeSpans().AppendEmpty();scope.Scope().SetName("hqsb.capture")
 expected:=make(map[int64]map[string]interface{}); starts:=make(map[int64]uint64)
 for i:=0;i<n;i++ {s:=scope.Spans().AppendEmpty();s.SetName(fmt.Sprintf("capture_%d",i%8));var id pcommon.SpanID;binary.BigEndian.PutUint64(id[:],uint64(i+1));s.SetSpanID(id)
  var tid pcommon.TraceID;binary.BigEndian.PutUint64(tid[8:],uint64(i+1));s.SetTraceID(tid)
  ts:=int64(1000000+i);start:=uint64(1000000000+i)
  if shape=="ns64" {ts=39719559540269779+int64(i);start=1792197494445574000+uint64(i)}
  if shape=="wide" {ts=int64(i);start=uint64(1000000+i)}
  body:=`{"name":"module_0","data":"`+strings.Repeat("x",64)+`"}`
  if shape=="wide" {body=`{"name":"module_`+strconv.Itoa(i)+`","data":"`+strings.Repeat("x",1024)+`"}`}
  s.SetStartTimestamp(pcommon.Timestamp(start));s.SetEndTimestamp(pcommon.Timestamp(start+17))
  a:=s.Attributes();a.PutStr("run_id","run_admission");a.PutInt("producer_id",int64(i%8));a.PutInt("source_seq",int64(i));a.PutInt("source_timestamp",ts);a.PutStr("body",body)
  expected[int64(i)]=a.AsRaw();starts[int64(i)]=start
 }
 settings:=receiver.CreateSettings{ID:component.NewID(component.MustNewType("hqsb")),TelemetrySettings:component.TelemetrySettings{Logger:zap.NewNop(),MeterProvider:metricnoop.NewMeterProvider(),TracerProvider:tracenoop.NewTracerProvider()}}
 obs,err:=receiverhelper.NewObsReport(receiverhelper.ObsReportSettings{ReceiverID:settings.ID,Transport:"http",ReceiverCreateSettings:settings});if err!=nil {t.Fatal(err)}
 got:=make(map[int64]map[string]interface{});gotStarts:=make(map[int64]uint64);gotEnds:=make(map[int64]uint64)
 next,err:=consumer.NewTraces(func(_ context.Context,data ptrace.Traces)error{
  for ri:=0;ri<data.ResourceSpans().Len();ri++ {ss:=data.ResourceSpans().At(ri).ScopeSpans();for si:=0;si<ss.Len();si++ {sp:=ss.At(si).Spans();for k:=0;k<sp.Len();k++ {s:=sp.At(k);v,_:=s.Attributes().Get("source_seq");i:=v.Int();got[i]=s.Attributes().AsRaw();gotStarts[i]=uint64(s.StartTimestamp());gotEnds[i]=uint64(s.EndTimestamp())}}};return nil
 });if err!=nil {t.Fatal(err)}
 target:=trace.New(next,obs)
 request:=ptraceotlp.NewExportRequestFromTraces(td)
 plain,err:=request.MarshalJSON();if err!=nil {t.Fatal(err)}
 outDir:=os.Getenv("HQSB_OUTPUT");if err=os.MkdirAll(outDir,0700);err!=nil {t.Fatal(err)}
 if err=os.WriteFile(filepath.Join(outDir,"input.json"),plain,0600);err!=nil {t.Fatal(err)}
 var before,after runtime.MemStats;runtime.GC();runtime.ReadMemStats(&before)
 startEncode:=time.Now();var packet,dictPacket []byte
 if mode=="otlp-json" {packet,err=request.MarshalJSON()}else{
  configs:=[][3]int{{256,16,1000},{1024,100,10000},{4096,256,100000},{30000,100,50000}};c:=configs[profile]
  id,full,increment,data:=request.MarshalWithTraceZip(c[0],c[1],c[2],false,os.Getenv("HQSB_DELETE_RESOURCE")=="1")
  packet,err=json.Marshal(map[string]interface{}{"_":id,"a":data})
  if len(full)>0 {dictPacket,err=json.Marshal(map[string]interface{}{"_":id,"t":"a","n":full})}else if len(increment)>0 {dictPacket,err=json.Marshal(map[string]interface{}{"_":id,"t":"i","n":increment})}
 }
 if err!=nil {t.Fatal(err)}
 zip:=mode=="tracezip-gzip"
 pack:=func(b []byte)[]byte{if !zip{return b};var buf bytes.Buffer;w:=gzip.NewWriter(&buf);if _,e:=w.Write(b);e!=nil {t.Fatal(e)};if e:=w.Close();e!=nil {t.Fatal(e)};return buf.Bytes()}
 packet=pack(packet);if len(dictPacket)>0 {dictPacket=pack(dictPacket)}
 encodeNs:=time.Since(startEncode).Nanoseconds();decodeStart:=time.Now()
 if len(dictPacket)>0 {req:=httptest.NewRequest("POST","/v1/traces-dict",bytes.NewReader(dictPacket));req.Header.Set("Content-Type","application/json");if zip {req.Header.Set("Content-Encoding","gzip")};res:=httptest.NewRecorder();handleTracesDictionary(res,req);if res.Code!=200 {t.Fatalf("dictionary HTTP %d: %s",res.Code,res.Body.String())}}
 req:=httptest.NewRequest("POST","/v1/traces",bytes.NewReader(packet));req.Header.Set("Content-Type","application/json");if zip {req.Header.Set("Content-Encoding","gzip")};res:=httptest.NewRecorder();handleTraces(res,req,target,"",mode=="otlp-json")
 decodeNs:=time.Since(decodeStart).Nanoseconds();runtime.ReadMemStats(&after)
 fieldErrors:=make(map[string]int);spanTimeErrors:=0;valid:=0
 for i,want:=range expected {have,ok:=got[i];if !ok {fieldErrors["missing_record"]++;continue};good:=true;for key,value:=range want {if !reflect.DeepEqual(value,have[key]) {fieldErrors[key]++;good=false}};if good {valid++};if gotStarts[i]!=starts[i] || gotEnds[i]!=starts[i]+17 {spanTimeErrors++}}
 output:=map[string]interface{}{"track":"TRACEZIP_COMPONENT_ADMISSION","mode":mode,"shape":shape,"profile":profile,"scheduled":n,"decoded_unique":len(got),"valid_unique_5fields":valid,"field_errors":fieldErrors,"span_timestamp_errors":spanTimeErrors,"http_code":res.Code,"plain_bytes":len(plain),"data_wire_bytes":len(packet),"dictionary_wire_bytes":len(dictPacket),"encode_ns":encodeNs,"decode_ns":decodeNs,"allocated_bytes":after.TotalAlloc-before.TotalAlloc,"allocations":after.Mallocs-before.Mallocs,"heap_after_bytes":after.HeapAlloc,"scope":"finite prebuilt batch; no HTTP sender bottleneck; in-process stock receiver handler; not durable collection throughput"}
 data,_:=json.MarshalIndent(output,"","  ");if e:=os.WriteFile(filepath.Join(outDir,"metrics.json"),data,0600);e!=nil {t.Fatal(e)}
 os.WriteFile(filepath.Join(outDir,"packet.bin"),packet,0600);os.WriteFile(filepath.Join(outDir,"dictionary.bin"),dictPacket,0600)
 decoded,_:=json.Marshal(got);os.WriteFile(filepath.Join(outDir,"decoded.json"),decoded,0600)
 // Keep diagnostics even on failed semantic admission; no performance ranking.
 if valid!=n || spanTimeErrors!=0 || res.Code!=200 {t.Errorf("semantic admission failed: valid=%d/%d, spanTimeErrors=%d, errors=%v",valid,n,spanTimeErrors,fieldErrors)}
}
