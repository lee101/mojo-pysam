"""Hot parsing kernels used by the Python compatibility layer."""

from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]


def find_byte(src: BPtr, begin: Int, end: Int, target: UInt8) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var i = begin
    var needle = SIMD[DType.uint8, W](target)
    while i + W <= end:
        var values = src.load[width=W](i)
        if values.eq(needle).reduce_or():
            for lane in range(W):
                if values[lane] == target:
                    return i + lane
        i += W
    while i < end:
        if src[i] == target:
            return i
        i += 1
    return end


def parse_uint(src: BPtr, begin: Int, end: Int) -> Int:
    if begin >= end:
        return -1
    var value = 0
    for i in range(begin, end):
        var c = Int(src[i])
        if c < 48 or c > 57:
            return -1
        value = value * 10 + c - 48
    return value


def parse_int(src: BPtr, begin: Int, end: Int) -> Int:
    if begin >= end:
        return 0
    var sign = 1
    var pos = begin
    if src[pos] == UInt8(45):
        sign = -1
        pos += 1
    var value = parse_uint(src, pos, end)
    return sign * value


def cigar_metrics_text(
    src: BPtr, begin: Int, end: Int, metrics: IPtr
):
    var ref_span = 0
    var query_span = 0
    var leading_soft = 0
    var trailing_soft = 0
    var number = 0
    var seen_alignment = False
    for i in range(begin, end):
        var c = Int(src[i])
        if c >= 48 and c <= 57:
            number = number * 10 + c - 48
            continue
        if c == 77 or c == 68 or c == 78 or c == 61 or c == 88:
            ref_span += number
        if c == 77 or c == 73 or c == 83 or c == 61 or c == 88:
            query_span += number
        if c == 83:
            if not seen_alignment:
                leading_soft += number
            trailing_soft += number
        elif c != 72 and c != 80:
            trailing_soft = 0
            seen_alignment = True
        number = 0
    metrics[0] = ref_span
    metrics[1] = query_span
    metrics[2] = leading_soft
    metrics[3] = trailing_soft


def load_u32(src: BPtr, pos: Int) -> Int:
    return (
        Int(src[pos])
        | (Int(src[pos + 1]) << 8)
        | (Int(src[pos + 2]) << 16)
        | (Int(src[pos + 3]) << 24)
    )


def load_i32(src: BPtr, pos: Int) -> Int:
    var value = load_u32(src, pos)
    if value >= 2147483648:
        return value - 4294967296
    return value


def cigar_metrics_bam(
    src: BPtr, begin: Int, count: Int, metrics: IPtr
):
    var ref_span = 0
    var query_span = 0
    var leading_soft = 0
    var trailing_soft = 0
    var seen_alignment = False
    for i in range(count):
        var packed = load_u32(src, begin + i * 4)
        var length = packed >> 4
        var op = packed & 15
        if op == 0 or op == 2 or op == 3 or op == 7 or op == 8:
            ref_span += length
        if op == 0 or op == 1 or op == 4 or op == 7 or op == 8:
            query_span += length
        if op == 4:
            if not seen_alignment:
                leading_soft += length
            trailing_soft += length
        elif op != 5 and op != 6:
            trailing_soft = 0
            seen_alignment = True
    metrics[0] = ref_span
    metrics[1] = query_span
    metrics[2] = leading_soft
    metrics[3] = trailing_soft


@export("mp_scan_sam")
def mp_scan_sam(
    data_addr: Int, size: Int, rows_addr: Int, capacity: Int
) abi("C") -> Int:
    if size < 0 or capacity < 0:
        return -3
    if size == 0:
        return 0
    if data_addr == 0 or rows_addr == 0 or capacity == 0:
        return -4
    var src = BPtr(unsafe_from_address=data_addr)
    var rows = IPtr(unsafe_from_address=rows_addr)
    var pos = 0
    var count = 0
    while pos < size:
        var line_begin = pos
        var line_end = find_byte(src, pos, size, UInt8(10))
        pos = line_end
        if line_end > line_begin and src[line_end - 1] == UInt8(13):
            line_end -= 1
        pos += 1
        if line_begin == line_end or src[line_begin] == UInt8(64):
            continue
        if count >= capacity:
            return -1

        var base = count * 24
        rows[base] = line_begin
        rows[base + 1] = line_end
        var field = 0
        var field_begin = line_begin
        while field_begin <= line_end:
            var i = find_byte(src, field_begin, line_end, UInt8(9))
            if field == 0:
                rows[base + 2] = i
            elif field == 1:
                rows[base + 3] = parse_uint(src, field_begin, i)
            elif field == 2:
                rows[base + 4] = field_begin
                rows[base + 5] = i
            elif field == 3:
                var sam_pos = parse_uint(src, field_begin, i)
                rows[base + 6] = sam_pos - 1 if sam_pos > 0 else -1
            elif field == 4:
                rows[base + 7] = parse_uint(src, field_begin, i)
            elif field == 5:
                rows[base + 8] = field_begin
                rows[base + 9] = i
            elif field == 6:
                rows[base + 10] = field_begin
                rows[base + 11] = i
            elif field == 7:
                var mate_pos = parse_uint(src, field_begin, i)
                rows[base + 12] = mate_pos - 1 if mate_pos > 0 else -1
            elif field == 8:
                rows[base + 13] = parse_int(src, field_begin, i)
            elif field == 9:
                rows[base + 14] = field_begin
                rows[base + 15] = i
            elif field == 10:
                rows[base + 16] = field_begin
                rows[base + 17] = i
                rows[base + 18] = i + 1 if i < line_end else line_end
            field += 1
            field_begin = i + 1
            if i == line_end:
                break
        if field < 11:
            return -2

        var metrics = IPtr(
            unsafe_from_address=rows_addr + (base + 19) * 8
        )
        cigar_metrics_text(
            src, rows[base + 8], rows[base + 9], metrics
        )
        rows[base + 23] = field
        count += 1
    return count


@export("mp_scan_bam")
def mp_scan_bam(
    data_addr: Int, size: Int, rows_addr: Int, capacity: Int
) abi("C") -> Int:
    if size < 0 or capacity < 0:
        return -5
    if size == 0:
        return 0
    if data_addr == 0 or rows_addr == 0 or capacity == 0:
        return -6
    var src = BPtr(unsafe_from_address=data_addr)
    var rows = IPtr(unsafe_from_address=rows_addr)
    var pos = 0
    var count = 0
    while pos < size:
        if pos + 36 > size:
            return -2
        var block_size = load_i32(src, pos)
        if block_size < 32 or block_size > size - pos - 4:
            return -3
        var record_end = pos + block_size + 4
        if count >= capacity:
            return -1
        var core = pos + 4
        var name_len = Int(src[core + 8])
        var mapq = Int(src[core + 9])
        var bin_value = Int(src[core + 10]) | (Int(src[core + 11]) << 8)
        var cigar_count = Int(src[core + 12]) | (Int(src[core + 13]) << 8)
        var flag = Int(src[core + 14]) | (Int(src[core + 15]) << 8)
        var seq_len = load_i32(src, core + 16)
        if name_len < 1 or seq_len < 0:
            return -4
        var name_begin = core + 32
        if name_len > record_end - name_begin:
            return -4
        var cigar_begin = name_begin + name_len
        if src[cigar_begin - 1] != UInt8(0):
            return -4
        if cigar_count > (record_end - cigar_begin) // 4:
            return -4
        var seq_begin = cigar_begin + cigar_count * 4
        if seq_len > (record_end - seq_begin) * 2:
            return -4
        var qual_begin = seq_begin + (seq_len + 1) // 2
        if seq_len > record_end - qual_begin:
            return -4
        var tags_begin = qual_begin + seq_len

        var base = count * 22
        rows[base] = pos
        rows[base + 1] = record_end
        rows[base + 2] = load_i32(src, core)
        rows[base + 3] = load_i32(src, core + 4)
        rows[base + 4] = name_len
        rows[base + 5] = mapq
        rows[base + 6] = bin_value
        rows[base + 7] = cigar_count
        rows[base + 8] = flag
        rows[base + 9] = seq_len
        rows[base + 10] = load_i32(src, core + 20)
        rows[base + 11] = load_i32(src, core + 24)
        rows[base + 12] = load_i32(src, core + 28)
        rows[base + 13] = name_begin
        rows[base + 14] = cigar_begin
        rows[base + 15] = seq_begin
        rows[base + 16] = qual_begin
        rows[base + 17] = tags_begin
        var metrics = IPtr(
            unsafe_from_address=rows_addr + (base + 18) * 8
        )
        cigar_metrics_bam(
            src, cigar_begin, cigar_count, metrics
        )
        count += 1
        pos = record_end
    return count


@export("mp_reg2bin")
def mp_reg2bin(begin: Int, end: Int) abi("C") -> Int:
    var stop = end - 1
    if begin >> 14 == stop >> 14:
        return 4681 + (begin >> 14)
    if begin >> 17 == stop >> 17:
        return 585 + (begin >> 17)
    if begin >> 20 == stop >> 20:
        return 73 + (begin >> 20)
    if begin >> 23 == stop >> 23:
        return 9 + (begin >> 23)
    if begin >> 26 == stop >> 26:
        return 1 + (begin >> 26)
    return 0


@export("mp_count_coverage")
def mp_count_coverage(
    starts_addr: Int,
    starts_len: Int,
    cigar_offsets_addr: Int,
    cigar_offsets_len: Int,
    cigar_addr: Int,
    cigar_len: Int,
    seq_offsets_addr: Int,
    seq_offsets_len: Int,
    seq_addr: Int,
    seq_len: Int,
    records: Int,
    target_begin: Int,
    target_end: Int,
    counts_addr: Int,
    counts_len: Int,
) abi("C") -> Int:
    if (
        records < 0
        or target_end < target_begin
        or starts_len < records
        or cigar_offsets_len < records + 1
        or seq_offsets_len < records + 1
        or cigar_len < 0
        or seq_len < 0
        or counts_len < 4 * (target_end - target_begin)
    ):
        return -1
    if records == 0 or target_end == target_begin:
        return 0
    if (
        starts_addr == 0
        or cigar_offsets_addr == 0
        or seq_offsets_addr == 0
        or counts_addr == 0
    ):
        return -2
    if cigar_len > 0 and cigar_addr == 0:
        return -2
    if seq_len > 0 and seq_addr == 0:
        return -2
    var starts = IPtr(unsafe_from_address=starts_addr)
    var cigar_offsets = IPtr(unsafe_from_address=cigar_offsets_addr)
    var cigars = U32Ptr(unsafe_from_address=cigar_addr)
    var seq_offsets = IPtr(unsafe_from_address=seq_offsets_addr)
    var seq = BPtr(unsafe_from_address=seq_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var width = target_end - target_begin
    for record in range(records):
        if (
            cigar_offsets[record] < 0
            or cigar_offsets[record] > cigar_offsets[record + 1]
            or cigar_offsets[record + 1] > cigar_len
            or seq_offsets[record] < 0
            or seq_offsets[record] > seq_offsets[record + 1]
            or seq_offsets[record + 1] > seq_len
        ):
            return -3
        var ref_pos = starts[record]
        var query_pos = seq_offsets[record]
        for ci in range(cigar_offsets[record], cigar_offsets[record + 1]):
            var packed = Int(cigars[ci])
            var length = packed >> 4
            var op = packed & 15
            if op == 0 or op == 7 or op == 8:
                if length > seq_offsets[record + 1] - query_pos:
                    return -4
                for j in range(length):
                    var rp = ref_pos + j
                    if rp >= target_begin and rp < target_end:
                        var base = Int(seq[query_pos + j])
                        if base < 4:
                            counts[base * width + rp - target_begin] += 1
                ref_pos += length
                query_pos += length
            elif op == 1 or op == 4:
                if length > seq_offsets[record + 1] - query_pos:
                    return -4
                query_pos += length
            elif op == 2 or op == 3:
                ref_pos += length
    return 0


@export("mp_count_coverage_bam")
def mp_count_coverage_bam(
    data_addr: Int,
    data_size: Int,
    rows_addr: Int,
    rows_len: Int,
    records: Int,
    target_ref: Int,
    target_begin: Int,
    target_end: Int,
    quality_threshold: Int,
    filter_flags: Int,
    counts_addr: Int,
    counts_len: Int,
) abi("C") -> Int:
    if (
        data_size < 0
        or records < 0
        or rows_len < records * 22
        or target_end < target_begin
        or counts_len < 4 * (target_end - target_begin)
    ):
        return -1
    if records == 0 or target_end == target_begin:
        return 0
    if data_addr == 0 or rows_addr == 0 or counts_addr == 0:
        return -2
    var src = BPtr(unsafe_from_address=data_addr)
    var rows = IPtr(unsafe_from_address=rows_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var width = target_end - target_begin
    for record in range(records):
        var base = record * 22
        var record_end = rows[base + 1]
        var cigar_begin = rows[base + 14]
        var cigar_count = rows[base + 7]
        var seq_begin = rows[base + 15]
        var qual_begin = rows[base + 16]
        var query_length = rows[base + 9]
        if (
            record_end < 0
            or record_end > data_size
            or cigar_begin < 0
            or cigar_count < 0
            or cigar_count > (record_end - cigar_begin) // 4
            or seq_begin < cigar_begin + cigar_count * 4
            or qual_begin < seq_begin
            or query_length < 0
            or query_length > record_end - qual_begin
            or (query_length + 1) // 2 > qual_begin - seq_begin
        ):
            return -3
        if rows[base + 2] != target_ref:
            continue
        if filter_flags != 0 and (rows[base + 8] & filter_flags) != 0:
            continue
        var ref_pos = rows[base + 3]
        if ref_pos >= target_end or ref_pos + rows[base + 18] <= target_begin:
            continue
        var query_pos = 0
        for ci in range(cigar_count):
            var packed = load_u32(src, cigar_begin + ci * 4)
            var length = packed >> 4
            var op = packed & 15
            if op == 0 or op == 7 or op == 8:
                if length > query_length - query_pos:
                    return -4
                for j in range(length):
                    var rp = ref_pos + j
                    if (
                        rp >= target_begin
                        and rp < target_end
                        and Int(src[qual_begin + query_pos + j]) >= quality_threshold
                    ):
                        var packed_base = Int(src[seq_begin + (query_pos + j) // 2])
                        var code = (
                            packed_base >> 4
                            if (query_pos + j) % 2 == 0
                            else packed_base & 15
                        )
                        var channel = -1
                        if code == 1:
                            channel = 0
                        elif code == 2:
                            channel = 1
                        elif code == 4:
                            channel = 2
                        elif code == 8:
                            channel = 3
                        if channel >= 0:
                            counts[channel * width + rp - target_begin] += 1
                ref_pos += length
                query_pos += length
            elif op == 1 or op == 4:
                if length > query_length - query_pos:
                    return -4
                query_pos += length
            elif op == 2 or op == 3:
                ref_pos += length
    return 0
