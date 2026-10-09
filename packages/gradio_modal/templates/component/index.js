const {
  SvelteComponent: Ts,
  assign: Cs,
  children: Ms,
  claim_element: Bs,
  create_slot: zs,
  detach: xn,
  element: Rs,
  get_all_dirty_from_scope: Ns,
  get_slot_changes: Is,
  get_spread_update: Ls,
  init: Os,
  insert_hydration: qs,
  safe_not_equal: Ps,
  set_dynamic_element_data: An,
  set_style: ze,
  toggle_class: a0,
  transition_in: Qi,
  transition_out: Ji,
  update_slot_base: Hs
} = window.__gradio__svelte__internal;


function Us(a) {
  let e, t, r;
  const n = (
    /*#slots*/
    a[22].default
  ), i = zs(
    n,
    a,
    /*$$scope*/
    a[21],
    null
  );
  let l = [
    { "data-testid": (
      /*test_id*/
      a[9]
    ) },
    { id: (
      /*elem_id*/
      a[4]
    ) },
    {
      class: t = "block " + /*elem_classes*/
      a[5].join(" ") + " svelte-1ezsyiy"
    }
  ], o = {};
  for (let u = 0; u < l.length; u += 1)
    o = Cs(o, l[u]);
  return {
    c() {
      e = Rs(
        /*tag*/
        a[18]
      ), i && i.c(), this.h();
    },
    l(u) {
      e = Bs(
        u,
        /*tag*/
        (a[18] || "null").toUpperCase(),
        {
          "data-testid": !0,
          id: !0,
          class: !0
        }
      );
      var h = Ms(e);
      i && i.l(h), h.forEach(xn), this.h();
    },
    h() {
      An(
        /*tag*/
        a[18]
      )(e, o), a0(
        e,
        "hidden",
        /*visible*/
        a[12] === !1
      ), a0(
        e,
        "padded",
        /*padding*/
        a[8]
      ), a0(
        e,
        "flex",
        /*flex*/
        a[17]
      ), a0(
        e,
        "border_focus",
        /*border_mode*/
        a[7] === "focus"
      ), a0(
        e,
        "border_contrast",
        /*border_mode*/
        a[7] === "contrast"
      ), a0(e, "hide-container", !/*explicit_call*/
      a[10] && !/*container*/
      a[11]), ze(
        e,
        "height",
        /*get_dimension*/
        a[19](
          /*height*/
          a[0]
        )
      ), ze(
        e,
        "min-height",
        /*get_dimension*/
        a[19](
          /*min_height*/
          a[1]
        )
      ), ze(
        e,
        "max-height",
        /*get_dimension*/
        a[19](
          /*max_height*/
          a[2]
        )
      ), ze(e, "width", typeof /*width*/
      a[3] == "number" ? `calc(min(${/*width*/
      a[3]}px, 100%))` : (
        /*get_dimension*/
        a[19](
          /*width*/
          a[3]
        )
      )), ze(
        e,
        "border-style",
        /*variant*/
        a[6]
      ), ze(
        e,
        "overflow",
        /*allow_overflow*/
        a[13] ? (
          /*overflow_behavior*/
          a[14]
        ) : "hidden"
      ), ze(
        e,
        "flex-grow",
        /*scale*/
        a[15]
      ), ze(e, "min-width", `calc(min(${/*min_width*/
      a[16]}px, 100%))`), ze(e, "border-width", "var(--block-border-width)");
    },
    m(u, h) {
      qs(u, e, h), i && i.m(e, null), r = !0;
    },
    p(u, h) {
      i && i.p && (!r || h & /*$$scope*/
      2097152) && Hs(
        i,
        n,
        u,
        /*$$scope*/
        u[21],
        r ? Is(
          n,
          /*$$scope*/
          u[21],
          h,
          null
        ) : Ns(
          /*$$scope*/
          u[21]
        ),
        null
      ), An(
        /*tag*/
        u[18]
      )(e, o = Ls(l, [
        (!r || h & /*test_id*/
        512) && { "data-testid": (
          /*test_id*/
          u[9]
        ) },
        (!r || h & /*elem_id*/
        16) && { id: (
          /*elem_id*/
          u[4]
        ) },
        (!r || h & /*elem_classes*/
        32 && t !== (t = "block " + /*elem_classes*/
        u[5].join(" ") + " svelte-1ezsyiy")) && { class: t }
      ])), a0(
        e,
        "hidden",
        /*visible*/
        u[12] === !1
      ), a0(
        e,
        "padded",
        /*padding*/
        u[8]
      ), a0(
        e,
        "flex",
        /*flex*/
        u[17]
      ), a0(
        e,
        "border_focus",
        /*border_mode*/
        u[7] === "focus"
      ), a0(
        e,
        "border_contrast",
        /*border_mode*/
        u[7] === "contrast"
      ), a0(e, "hide-container", !/*explicit_call*/
      u[10] && !/*container*/
      u[11]), h & /*height*/
      1 && ze(
        e,
        "height",
        /*get_dimension*/
        u[19](
          /*height*/
          u[0]
        )
      ), h & /*min_height*/
      2 && ze(
        e,
        "min-height",
        /*get_dimension*/
        u[19](
          /*min_height*/
          u[1]
        )
      ), h & /*max_height*/
      4 && ze(
        e,
        "max-height",
        /*get_dimension*/
        u[19](
          /*max_height*/
          u[2]
        )
      ), h & /*width*/
      8 && ze(e, "width", typeof /*width*/
      u[3] == "number" ? `calc(min(${/*width*/
      u[3]}px, 100%))` : (
        /*get_dimension*/
        u[19](
          /*width*/
          u[3]
        )
      )), h & /*variant*/
      64 && ze(
        e,
        "border-style",
        /*variant*/
        u[6]
      ), h & /*allow_overflow, overflow_behavior*/
      24576 && ze(
        e,
        "overflow",
        /*allow_overflow*/
        u[13] ? (
          /*overflow_behavior*/
          u[14]
        ) : "hidden"
      ), h & /*scale*/
      32768 && ze(
        e,
        "flex-grow",
        /*scale*/
        u[15]
      ), h & /*min_width*/
      65536 && ze(e, "min-width", `calc(min(${/*min_width*/
      u[16]}px, 100%))`);
    },
    i(u) {
      r || (Qi(i, u), r = !0);
    },
    o(u) {
      Ji(i, u), r = !1;
    },
    d(u) {
      u && xn(e), i && i.d(u);
    }
  };
}

function Vs(a) {
  let e, t = (
    /*tag*/
    a[18] && Us(a)
  );
  return {
    c() {
      t && t.c();
    },
    l(r) {
      t && t.l(r);
    },
    m(r, n) {
      t && t.m(r, n), e = !0;
    },
    p(r, [n]) {
      /*tag*/
      r[18] && t.p(r, n);
    },
    i(r) {
      e || (Qi(t, r), e = !0);
    },
    o(r) {
      Ji(t, r), e = !1;
    },
    d(r) {
      t && t.d(r);
    }
  };
}

function Gs(a, e, t) {
  let { $$slots: r = {}, $$scope: n } = e, { height: i = void 0 } = e, { min_height: l = void 0 } = e, { max_height: o = void 0 } = e, { width: u = void 0 } = e, { elem_id: h = "" } = e, { elem_classes: f = [] } = e, { variant: p = "solid" } = e, { border_mode: v = "base" } = e, { padding: w = !0 } = e, { type: E = "normal" } = e, { test_id: F = void 0 } = e, { explicit_call: _ = !1 } = e, { container: T = !0 } = e, { visible: D = !0 } = e, { allow_overflow: y = !0 } = e, { overflow_behavior: A = "auto" } = e, { scale: C = null } = e, { min_width: M = 0 } = e, { flex: B = !1 } = e, H = E === "fieldset" ? "fieldset" : "div";
  const N = (I) => {
    if (I !== void 0) {
      if (typeof I == "number")
        return I + "px";
      if (typeof I == "string")
        return I;
    }
  };
  return a.$$set = (I) => {
    "height" in I && t(0, i = I.height), "min_height" in I && t(1, l = I.min_height), "max_height" in I && t(2, o = I.max_height), "width" in I && t(3, u = I.width), "elem_id" in I && t(4, h = I.elem_id), "elem_classes" in I && t(5, f = I.elem_classes), "variant" in I && t(6, p = I.variant), "border_mode" in I && t(7, v = I.border_mode), "padding" in I && t(8, w = I.padding), "type" in I && t(20, E = I.type), "test_id" in I && t(9, F = I.test_id), "explicit_call" in I && t(10, _ = I.explicit_call), "container" in I && t(11, T = I.container), "visible" in I && t(12, D = I.visible), "allow_overflow" in I && t(13, y = I.allow_overflow), "overflow_behavior" in I && t(14, A = I.overflow_behavior), "scale" in I && t(15, C = I.scale), "min_width" in I && t(16, M = I.min_width), "flex" in I && t(17, B = I.flex), "$$scope" in I && t(21, n = I.$$scope);
  }, [
    i,
    l,
    o,
    u,
    h,
    f,
    p,
    v,
    w,
    F,
    _,
    T,
    D,
    y,
    A,
    C,
    M,
    B,
    H,
    N,
    E,
    n,
    r
  ];
}
class Ws extends Ts {
  constructor(e) {
    super(), Os(this, e, Gs, Vs, Ps, {
      height: 0,
      min_height: 1,
      max_height: 2,
      width: 3,
      elem_id: 4,
      elem_classes: 5,
      variant: 6,
      border_mode: 7,
      padding: 8,
      type: 20,
      test_id: 9,
      explicit_call: 10,
      container: 11,
      visible: 12,
      allow_overflow: 13,
      overflow_behavior: 14,
      scale: 15,
      min_width: 16,
      flex: 17
    });
  }
}


const {
  SvelteComponent: Y4,
  append_hydration: j4,
  assign: X4,
  attr: dr,
  check_outros: Z4,
  children: K4,
  claim_component: Q4,
  claim_element: J4,
  claim_space: $4,
  create_component: ec,
  create_slot: tc,
  destroy_component: rc,
  detach: Wi,
  element: ac,
  get_all_dirty_from_scope: nc,
  get_slot_changes: ic,
  get_spread_object: lc,
  get_spread_update: sc,
  group_outros: oc,
  init: uc,
  insert_hydration: cc,
  mount_component: hc,
  safe_not_equal: mc,
  set_style: pr,
  space: fc,
  toggle_class: J0,
  transition_in: Lt,
  transition_out: wr,
  update_slot_base: dc
} = window.__gradio__svelte__internal;
function Yi(a) {
  let e, t;
  const r = [
    { autoscroll: (
      /*gradio*/
      a[8].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      a[8].i18n
    ) },
    /*loading_status*/
    a[7],
    {
      status: (
        /*loading_status*/
        a[7] ? (
          /*loading_status*/
          a[7].status == "pending" ? "generating" : (
            /*loading_status*/
            a[7].status
          )
        ) : null
      )
    }
  ];
  let n = {};
  for (let i = 0; i < r.length; i += 1)
    n = X4(n, r[i]);
  return e = new c4({ props: n }), {
    c() {
      ec(e.$$.fragment);
    },
    l(i) {
      Q4(e.$$.fragment, i);
    },
    m(i, l) {
      hc(e, i, l), t = !0;
    },
    p(i, l) {
      const o = l & /*gradio, loading_status*/
      384 ? sc(r, [
        l & /*gradio*/
        256 && { autoscroll: (
          /*gradio*/
          i[8].autoscroll
        ) },
        l & /*gradio*/
        256 && { i18n: (
          /*gradio*/
          i[8].i18n
        ) },
        l & /*loading_status*/
        128 && lc(
          /*loading_status*/
          i[7]
        ),
        l & /*loading_status*/
        128 && {
          status: (
            /*loading_status*/
            i[7] ? (
              /*loading_status*/
              i[7].status == "pending" ? "generating" : (
                /*loading_status*/
                i[7].status
              )
            ) : null
          )
        }
      ]) : {};
      e.$set(o);
    },
    i(i) {
      t || (Lt(e.$$.fragment, i), t = !0);
    },
    o(i) {
      wr(e.$$.fragment, i), t = !1;
    },
    d(i) {
      rc(e, i);
    }
  };
}
function pc(a) {
  let e, t, r, n = `calc(min(${/*min_width*/
  a[2]}px, 100%))`, i, l = (
    /*loading_status*/
    a[7] && /*show_progress*/
    a[9] && /*gradio*/
    a[8] && Yi(a)
  );
  const o = (
    /*#slots*/
    a[11].default
  ), u = tc(
    o,
    a,
    /*$$scope*/
    a[10],
    null
  );
  return {
    c() {
      e = ac("div"), l && l.c(), t = fc(), u && u.c(), this.h();
    },
    l(h) {
      e = J4(h, "DIV", { id: !0, class: !0 });
      var f = K4(e);
      l && l.l(f), t = $4(f), u && u.l(f), f.forEach(Wi), this.h();
    },
    h() {
      dr(
        e,
        "id",
        /*elem_id*/
        a[3]
      ), dr(e, "class", r = "column " + /*elem_classes*/
      a[4].join(" ") + " svelte-1m1obck"), J0(
        e,
        "gap",
        /*gap*/
        a[1]
      ), J0(
        e,
        "compact",
        /*variant*/
        a[6] === "compact"
      ), J0(
        e,
        "panel",
        /*variant*/
        a[6] === "panel"
      ), J0(e, "hide", !/*visible*/
      a[5]), pr(
        e,
        "flex-grow",
        /*scale*/
        a[0]
      ), pr(e, "min-width", n);
    },
    m(h, f) {
      cc(h, e, f), l && l.m(e, null), j4(e, t), u && u.m(e, null), i = !0;
    },
    p(h, [f]) {
      /*loading_status*/
      h[7] && /*show_progress*/
      h[9] && /*gradio*/
      h[8] ? l ? (l.p(h, f), f & /*loading_status, show_progress, gradio*/
      896 && Lt(l, 1)) : (l = Yi(h), l.c(), Lt(l, 1), l.m(e, t)) : l && (oc(), wr(l, 1, 1, () => {
        l = null;
      }), Z4()), u && u.p && (!i || f & /*$$scope*/
      1024) && dc(
        u,
        o,
        h,
        /*$$scope*/
        h[10],
        i ? ic(
          o,
          /*$$scope*/
          h[10],
          f,
          null
        ) : nc(
          /*$$scope*/
          h[10]
        ),
        null
      ), (!i || f & /*elem_id*/
      8) && dr(
        e,
        "id",
        /*elem_id*/
        h[3]
      ), (!i || f & /*elem_classes*/
      16 && r !== (r = "column " + /*elem_classes*/
      h[4].join(" ") + " svelte-1m1obck")) && dr(e, "class", r), (!i || f & /*elem_classes, gap*/
      18) && J0(
        e,
        "gap",
        /*gap*/
        h[1]
      ), (!i || f & /*elem_classes, variant*/
      80) && J0(
        e,
        "compact",
        /*variant*/
        h[6] === "compact"
      ), (!i || f & /*elem_classes, variant*/
      80) && J0(
        e,
        "panel",
        /*variant*/
        h[6] === "panel"
      ), (!i || f & /*elem_classes, visible*/
      48) && J0(e, "hide", !/*visible*/
      h[5]), f & /*scale*/
      1 && pr(
        e,
        "flex-grow",
        /*scale*/
        h[0]
      ), f & /*min_width*/
      4 && n !== (n = `calc(min(${/*min_width*/
      h[2]}px, 100%))`) && pr(e, "min-width", n);
    },
    i(h) {
      i || (Lt(l), Lt(u, h), i = !0);
    },
    o(h) {
      wr(l), wr(u, h), i = !1;
    },
    d(h) {
      h && Wi(e), l && l.d(), u && u.d(h);
    }
  };
}
function gc(a, e, t) {
  let { $$slots: r = {}, $$scope: n } = e, { scale: i = null } = e, { gap: l = !0 } = e, { min_width: o = 0 } = e, { elem_id: u = "" } = e, { elem_classes: h = [] } = e, { visible: f = !0 } = e, { variant: p = "default" } = e, { loading_status: v = void 0 } = e, { gradio: w = void 0 } = e, { show_progress: E = !1 } = e;
  return a.$$set = (F) => {
    "scale" in F && t(0, i = F.scale), "gap" in F && t(1, l = F.gap), "min_width" in F && t(2, o = F.min_width), "elem_id" in F && t(3, u = F.elem_id), "elem_classes" in F && t(4, h = F.elem_classes), "visible" in F && t(5, f = F.visible), "variant" in F && t(6, p = F.variant), "loading_status" in F && t(7, v = F.loading_status), "gradio" in F && t(8, w = F.gradio), "show_progress" in F && t(9, E = F.show_progress), "$$scope" in F && t(10, n = F.$$scope);
  }, [
    i,
    l,
    o,
    u,
    h,
    f,
    p,
    v,
    w,
    E,
    n,
    r
  ];
}

let vc = class extends Y4 {
  constructor(e) {
    super(), uc(this, e, gc, pc, mc, {
      scale: 0,
      gap: 1,
      min_width: 2,
      elem_id: 3,
      elem_classes: 4,
      visible: 5,
      variant: 6,
      loading_status: 7,
      gradio: 8,
      show_progress: 9
    });
  }
};
const {
  SvelteComponent: bc,
  append_hydration: yc,
  attr: bt,
  binding_callbacks: ji,
  children: Xi,
  claim_component: gs,
  claim_element: Fa,
  claim_space: wc,
  create_component: vs,
  create_slot: kc,
  destroy_component: bs,
  detach: Ut,
  element: _a,
  get_all_dirty_from_scope: Dc,
  get_slot_changes: xc,
  get_svelte_dataset: Ac,
  init: Fc,
  insert_hydration: Ja,
  listen: ys,
  mount_component: ws,
  noop: _c,
  safe_not_equal: Sc,
  space: Ec,
  toggle_class: Zi,
  transition_in: $a,
  transition_out: en,
  update_slot_base: Tc
} = window.__gradio__svelte__internal;


function Ki(a) {
  let e, t = '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 1L9 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9 1L1 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>', r, n;
  return {
    c() {
      e = _a("div"), e.innerHTML = t, this.h();
    },
    l(i) {
      e = Fa(i, "DIV", { class: !0, "data-svelte-h": !0 }), Ac(e) !== "svelte-a1nk6l" && (e.innerHTML = t), this.h();
    },
    h() {
      bt(e, "class", "close svelte-7knbu5");
    },
    m(i, l) {
      Ja(i, e, l), r || (n = ys(
        e,
        "click",
        /*close*/
        a[6]
      ), r = !0);
    },
    p: _c,
    d(i) {
      i && Ut(e), r = !1, n();
    }
  };
}
function Cc(a) {
  let e;
  const t = (
    /*#slots*/
    a[8].default
  ), r = kc(
    t,
    a,
    /*$$scope*/
    a[12],
    null
  );
  return {
    c() {
      r && r.c();
    },
    l(n) {
      r && r.l(n);
    },
    m(n, i) {
      r && r.m(n, i), e = !0;
    },
    p(n, i) {
      r && r.p && (!e || i & /*$$scope*/
      4096) && Tc(
        r,
        t,
        n,
        /*$$scope*/
        n[12],
        e ? xc(
          t,
          /*$$scope*/
          n[12],
          i,
          null
        ) : Dc(
          /*$$scope*/
          n[12]
        ),
        null
      );
    },
    i(n) {
      e || ($a(r, n), e = !0);
    },
    o(n) {
      en(r, n), e = !1;
    },
    d(n) {
      r && r.d(n);
    }
  };
}
function Mc(a) {
  let e, t, r, n = (
    /*allow_user_close*/
    a[3] && Ki(a)
  );
  return t = new vc({
    props: {
      $$slots: { default: [Cc] },
      $$scope: { ctx: a }
    }
  }), {
    c() {
      n && n.c(), e = Ec(), vs(t.$$.fragment);
    },
    l(i) {
      n && n.l(i), e = wc(i), gs(t.$$.fragment, i);
    },
    m(i, l) {
      n && n.m(i, l), Ja(i, e, l), ws(t, i, l), r = !0;
    },
    p(i, l) {
      /*allow_user_close*/
      i[3] ? n ? n.p(i, l) : (n = Ki(i), n.c(), n.m(e.parentNode, e)) : n && (n.d(1), n = null);
      const o = {};
      l & /*$$scope*/
      4096 && (o.$$scope = { dirty: l, ctx: i }), t.$set(o);
    },
    i(i) {
      r || ($a(t.$$.fragment, i), r = !0);
    },
    o(i) {
      en(t.$$.fragment, i), r = !1;
    },
    d(i) {
      i && Ut(e), n && n.d(i), bs(t, i);
    }
  };
}
function Bc(a) {
  let e, t, r, n, i, l, o;
  return r = new Ws({
    props: {
      allow_overflow: !1,
      elem_classes: ["modal-block"],
      $$slots: { default: [Mc] },
      $$scope: { ctx: a }
    }
  }), {
    c() {
      e = _a("div"), t = _a("div"), vs(r.$$.fragment), this.h();
    },
    l(u) {
      e = Fa(u, "DIV", { class: !0, id: !0 });
      var h = Xi(e);
      t = Fa(h, "DIV", { class: !0 });
      var f = Xi(t);
      gs(r.$$.fragment, f), f.forEach(Ut), h.forEach(Ut), this.h();
    },
    h() {
      bt(t, "class", "modal-container svelte-7knbu5"), bt(e, "class", n = "modal " + /*elem_classes*/
      a[2].join(" ") + " svelte-7knbu5"), bt(
        e,
        "id",
        /*elem_id*/
        a[1]
      ), Zi(e, "hide", !/*visible*/
      a[0]);
    },
    m(u, h) {
      Ja(u, e, h), yc(e, t), ws(r, t, null), a[9](t), a[10](e), i = !0, l || (o = ys(
        e,
        "click",
        /*click_handler*/
        a[11]
      ), l = !0);
    },
    p(u, [h]) {
      const f = {};
      h & /*$$scope, allow_user_close*/
      4104 && (f.$$scope = { dirty: h, ctx: u }), r.$set(f), (!i || h & /*elem_classes*/
      4 && n !== (n = "modal " + /*elem_classes*/
      u[2].join(" ") + " svelte-7knbu5")) && bt(e, "class", n), (!i || h & /*elem_id*/
      2) && bt(
        e,
        "id",
        /*elem_id*/
        u[1]
      ), (!i || h & /*elem_classes, visible*/
      5) && Zi(e, "hide", !/*visible*/
      u[0]);
    },
    i(u) {
      i || ($a(r.$$.fragment, u), i = !0);
    },
    o(u) {
      en(r.$$.fragment, u), i = !1;
    },
    d(u) {
      u && Ut(e), bs(r), a[9](null), a[10](null), l = !1, o();
    }
  };
}
function zc(a, e, t) {
  let { $$slots: r = {}, $$scope: n } = e, { elem_id: i = "" } = e, { elem_classes: l = [] } = e, { visible: o = !1 } = e, { allow_user_close: u = !0 } = e, { gradio: h } = e, f = null, p = null;
  const v = () => {
    t(0, o = !1), h.dispatch("blur");
  };
  document.addEventListener("keydown", (_) => {
    u && _.key === "Escape" && v();
  });
  function w(_) {
    ji[_ ? "unshift" : "push"](() => {
      p = _, t(5, p);
    });
  }
  function E(_) {
    ji[_ ? "unshift" : "push"](() => {
      f = _, t(4, f);
    });
  }
  const F = (_) => {
    u && (_.target === f || _.target === p) && v();
  };
  return a.$$set = (_) => {
    "elem_id" in _ && t(1, i = _.elem_id), "elem_classes" in _ && t(2, l = _.elem_classes), "visible" in _ && t(0, o = _.visible), "allow_user_close" in _ && t(3, u = _.allow_user_close), "gradio" in _ && t(7, h = _.gradio), "$$scope" in _ && t(12, n = _.$$scope);
  }, [
    o,
    i,
    l,
    u,
    f,
    p,
    v,
    h,
    r,
    w,
    E,
    F,
    n
  ];
}
class Nc extends bc {
  constructor(e) {
    super(), Fc(this, e, zc, Bc, Sc, {
      elem_id: 1,
      elem_classes: 2,
      visible: 0,
      allow_user_close: 3,
      gradio: 7
    });
  }
}
export {
  Nc as default
};
